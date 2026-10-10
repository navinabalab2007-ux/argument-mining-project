"""BERT baseline: one shared encoder with two heads (argument type + effectiveness).

Trained on exactly the same training split as the Llama fine-tuning so the comparison is fair.
"""
from pathlib import Path

import config
from utils import get_logger, load_json, now_iso, save_json

log = get_logger()


def _make_model_class():
    import torch.nn as nn
    from transformers import AutoModel

    class TwoHeadBert(nn.Module):
        def __init__(self, name, n_type, n_eff):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.type_head = nn.Linear(h, n_type)
            self.eff_head = nn.Linear(h, n_eff)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]
            pooled = self.drop(pooled)
            return self.type_head(pooled), self.eff_head(pooled)
    return TwoHeadBert


class BertBaseline:
    is_llm = False

    def __init__(self, model_name=config.BERT_MODEL_NAME, save_dir=config.BERT_DIR):
        self.model_name, self.save_dir = model_name, Path(save_dir)
        self.model = self.tokenizer = None
        self.type_labels = list(config.DISCOURSE_TYPES)
        self.eff_labels = list(config.EFFECTIVENESS_LABELS)
        self.meta = None

    @property
    def name(self):
        return f"{self.model_name} (fine-tuned BERT baseline)"

    kind = "bert"
    model_id = property(lambda self: self.model_name)

    def _device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_df, epochs=2, batch_size=8, lr=3e-5, max_length=128,
              progress_cb=None, seed=config.SEED):
        from model import require_packages, translate_error
        require_packages(["torch", "transformers"])
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        torch.manual_seed(seed)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = _make_model_class()(self.model_name, len(self.type_labels), len(self.eff_labels))
        except Exception as exc:
            raise translate_error(exc, self.model_name) from exc
        dev = self._device()
        self.model.to(dev)
        texts = list(train_df["discourse_text"])
        y_t = [self.type_labels.index(t) for t in train_df["discourse_type"]]
        y_e = [self.eff_labels.index(e) for e in train_df["discourse_effectiveness"]]
        n = len(texts)
        steps = max(1, epochs * ((n + batch_size - 1) // batch_size))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        ce = torch.nn.CrossEntropyLoss()
        self.model.train()
        g = torch.Generator().manual_seed(seed)
        losses, step = [], 0
        for ep in range(epochs):
            order = torch.randperm(n, generator=g).tolist()
            for b in range(0, n, batch_size):
                idx = order[b:b + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                     max_length=max_length, return_tensors="pt").to(dev)
                lt, le = self.model(enc["input_ids"], enc["attention_mask"])
                loss = ce(lt, torch.tensor([y_t[i] for i in idx], device=dev)) + \
                    ce(le, torch.tensor([y_e[i] for i in idx], device=dev))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                losses.append(float(loss.item())); step += 1
                if progress_cb:
                    progress_cb(step / steps, f"epoch {ep + 1}/{epochs} step {step}/{steps} loss {losses[-1]:.4f}")
        self.model.eval()
        self.meta = {"trained": True, "model_name": self.model_name, "n_train": n, "epochs": epochs,
                     "batch_size": batch_size, "lr": lr, "max_length": max_length, "steps": step,
                     "final_loss": losses[-1] if losses else None, "trained_at": now_iso(),
                     "type_labels": self.type_labels, "eff_labels": self.eff_labels}
        return self.meta

    def save(self):
        import torch
        if self.model is None or not self.meta:
            raise RuntimeError("Nothing to save: the BERT baseline has not been trained.")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_dir / "model.pt")
        self.tokenizer.save_pretrained(self.save_dir)
        save_json(self.meta, self.save_dir / "meta.json")
        return self.save_dir

    @classmethod
    def load(cls, save_dir=config.BERT_DIR):
        from model import require_packages
        require_packages(["torch", "transformers"])
        import torch
        from transformers import AutoTokenizer
        save_dir = Path(save_dir)
        meta = load_json(save_dir / "meta.json")
        if not meta or not meta.get("trained") or not (save_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"No trained BERT baseline found in '{save_dir}'. Run train_bert_sample.bat "
                f"(or 'python train.py --target bert') first.")
        obj = cls(meta["model_name"], save_dir)
        obj.meta, obj.type_labels, obj.eff_labels = meta, meta["type_labels"], meta["eff_labels"]
        obj.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        obj.model = _make_model_class()(meta["model_name"], len(obj.type_labels), len(obj.eff_labels))
        obj.model.load_state_dict(torch.load(save_dir / "model.pt", map_location="cpu"))
        obj.model.to(obj._device()).eval()
        return obj

    def classify_texts(self, texts, progress_cb=None, batch_size=16, max_length=128):
        import torch
        dev, out = self._device(), []
        with torch.no_grad():
            for b in range(0, len(texts), batch_size):
                chunk = texts[b:b + batch_size]
                enc = self.tokenizer(chunk, truncation=True, padding=True, max_length=max_length,
                                     return_tensors="pt").to(dev)
                lt, le = self.model(enc["input_ids"], enc["attention_mask"])
                for t, e in zip(lt.argmax(-1).tolist(), le.argmax(-1).tolist()):
                    out.append({"type": self.type_labels[t], "effectiveness": self.eff_labels[e]})
                if progress_cb:
                    progress_cb(min(1.0, (b + len(chunk)) / len(texts)), "BERT predicting")
        return out
