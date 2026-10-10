"""Fine-tuning: Llama 3.1 8B with LoRA/QLoRA (default) or the BERT baseline (--target bert).

Examples (run.bat-style helper files are included):
    python train.py --sample-mode                       # tiny smoke-test run on the built-in sample CSV
    python train.py --dataset feedback-prize --dataset-dir D:\\data\\feedback-prize --max-train-samples 2000
    python train.py --target bert --sample-mode

The adapter is only written after training really runs, together with training_meta.json.
Nothing is called 'fine-tuned' before that file exists.
"""
import argparse
import math
import sys
import time

import config
from data_loader import DataLoaderError, get_dataset, split_dataset, subsample
from model import ModelError, load_base_model, require_packages, translate_error
from preprocessing import build_classification_messages, format_target
from utils import ensure_dirs, get_logger, now_iso, save_json, truncate_words

log = get_logger()


def build_examples(df, tokenizer, max_length):
    """Tokenise (prompt -> 'Type: X / Quality: Y') pairs; loss is computed on the answer only."""
    examples = []
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    for _, row in df.iterrows():
        text = truncate_words(row["discourse_text"], config.MAX_SEGMENT_WORDS_FOR_MODEL)
        msgs = build_classification_messages(text, few_shot=False)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(format_target(row["discourse_type"], row["discourse_effectiveness"]),
                          add_special_tokens=False)["input_ids"]
        t_ids = t_ids + [eot if isinstance(eot, int) and eot >= 0 else tokenizer.eos_token_id]
        if len(p_ids) + len(t_ids) > max_length:
            continue  # skip over-long examples instead of cutting off the answer
        examples.append({"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids})
    return examples


def collate(batch, pad_id):
    import torch
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def prepare_training_frame(args):
    source = "sample" if args.sample_mode and args.dataset == "sample" else args.dataset
    df, rep = get_dataset(source, args.dataset_dir)
    train_df, _, _ = split_dataset(df, seed=args.seed)
    if train_df.empty:
        train_df = df
    limit = args.max_train_samples if args.max_train_samples else (48 if args.sample_mode else None)
    train_df = subsample(train_df, limit, args.seed)
    return train_df, rep


def run_llama_finetune(args):
    require_packages(["torch", "transformers", "accelerate", "peft"])
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader

    train_df, rep = prepare_training_frame(args)
    if rep["source"] == "sample":
        log.warning("Using the tiny built-in SAMPLE data: this only tests that the pipeline runs.")
    log.info("Training examples: %d (source=%s)", len(train_df), rep["source"])
    model, tok, notes = load_base_model(args.model_id, args.quant, args.hf_token, args.force, for_training=True)
    for n in notes:
        log.info(n)
    examples = build_examples(train_df, tok, args.max_length)
    if not examples:
        raise ValueError("No training examples fit within --max-length. Increase --max-length.")
    cuda = torch.cuda.is_available()
    is_4bit = getattr(model, "is_loaded_in_4bit", False)
    if is_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                      target_modules=config.LORA_TARGET_MODULES, task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    amp_dtype = torch.bfloat16 if (not cuda or torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(cuda and amp_dtype == torch.float16))
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, tok.pad_token_id),
                        generator=torch.Generator().manual_seed(args.seed))
    total_updates = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    model.train()
    losses, update, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=amp_dtype):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.item()) * args.grad_accum)
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                update += 1
                log.info("epoch %d/%d update %d/%d loss %.4f", ep + 1, args.epochs, update,
                         total_updates, losses[-1])
        ckpt = config.CHECKPOINTS_DIR / f"llama_lora_epoch{ep + 1}"
        model.save_pretrained(str(ckpt))
        log.info("Checkpoint saved: %s", ckpt)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    meta = {"trained": True, "base_model_id": args.model_id, "n_train_examples": len(examples),
            "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "lr": args.lr, "max_length": args.max_length, "lora_r": args.lora_r,
            "quantization": "4bit" if is_4bit else "none", "dataset_source": rep["source"],
            "sample_mode": bool(args.sample_mode), "updates": update,
            "first_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
            "seconds": round(time.time() - t0, 1), "trained_at": now_iso()}
    save_json(meta, out / "training_meta.json")
    return meta


def run_bert(args):
    from baseline import BertBaseline
    train_df, rep = prepare_training_frame(args)
    if rep["source"] == "sample":
        log.warning("Using the tiny built-in SAMPLE data: this only tests that the pipeline runs.")
    b = BertBaseline()
    meta = b.train(train_df, epochs=args.epochs, batch_size=max(2, args.batch_size), lr=args.lr,
                   max_length=min(args.max_length, 256),
                   progress_cb=lambda f, m: print(f"\r{m}", end="", flush=True), seed=args.seed)
    print()
    meta["dataset_source"] = rep["source"]
    b.meta = meta
    b.save()
    return meta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B (LoRA/QLoRA) or the BERT baseline.")
    ap.add_argument("--target", choices=["llama", "bert"], default="llama")
    ap.add_argument("--dataset", choices=["sample", "feedback-prize"], default="feedback-prize")
    ap.add_argument("--dataset-dir", default=None, help="Folder containing train.csv")
    ap.add_argument("--sample-mode", action="store_true",
                    help="Small test run (uses the built-in sample data unless --dataset feedback-prize)")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=config.LORA_R)
    ap.add_argument("--model-id", default=config.DEFAULT_MODEL_ID)
    ap.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto")
    ap.add_argument("--hf-token", default=None, help="Optional; otherwise HF_TOKEN / huggingface-cli login is used")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--force", action="store_true", help="Try even if memory looks too small")
    args = ap.parse_args(argv)
    if args.sample_mode and args.dataset_dir is None and args.dataset == "feedback-prize":
        try:  # sample mode without a downloaded dataset -> fall back to the built-in sample
            get_dataset("feedback-prize", None)
        except DataLoaderError:
            args.dataset = "sample"
    args.epochs = args.epochs or (1 if args.sample_mode else 2)
    args.lr = args.lr or (2e-4 if args.target == "llama" else 3e-5)
    from pathlib import Path
    args.output_dir = Path(args.output_dir) if args.output_dir else config.LLAMA_ADAPTER_DIR
    return args


def main(argv=None):
    ensure_dirs()
    args = parse_args(argv)
    try:
        meta = run_bert(args) if args.target == "bert" else run_llama_finetune(args)
    except (ModelError, DataLoaderError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        err = translate_error(exc, args.model_id)
        print(f"\nERROR: {err}")
        if "memory" in str(err).lower():
            print("Tips: --batch-size 1, --max-length 256, --quant 4bit (NVIDIA GPU), or a smaller --model-id.")
        return 1
    print("\nTraining finished. Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("\nNext: run evaluate_all.bat or open the app and choose the fine-tuned engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
