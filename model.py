"""Model loading: heuristic demo engine, pretrained Llama, and LoRA fine-tuned Llama.

torch / transformers / peft are imported lazily so the demo engine works even if they are missing.
"""
import contextlib
import importlib.util
import re
from pathlib import Path

import config
from utils import get_logger, hardware_summary, load_json

log = get_logger()


class ModelError(RuntimeError):
    """Base class for friendly model errors."""


class ModelDependencyError(ModelError):
    pass


class ModelAccessError(ModelError):
    pass


class ModelDownloadError(ModelError):
    pass


class ModelResourceError(ModelError):
    pass


def _has(pkg):
    return importlib.util.find_spec(pkg) is not None


def require_packages(names):
    missing = [n for n in names if not _has(n)]
    if missing:
        raise ModelDependencyError(
            f"Missing Python packages: {', '.join(missing)}. Run run.bat once (it installs everything) "
            f"or: pip install -r requirements.txt")


def resolve_quantization(quant, cuda):
    """Return ('4bit'|'none', note)."""
    if quant == "none":
        return "none", None
    if not cuda:
        return "none", "4-bit loading needs an NVIDIA GPU; loading in normal precision on CPU."
    if not _has("bitsandbytes"):
        if quant == "4bit":
            return "none", "bitsandbytes is not installed, so 4-bit loading is unavailable."
        return "none", None
    return "4bit", None


def _size_in_billions(model_id):
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", str(model_id).lower())
    return float(m.group(1)) if m else 8.0


def check_memory(model_id, quant_resolved, force=False):
    """Warn (or raise) when the machine is clearly too small for the model."""
    hw = hardware_summary()
    size_b = _size_in_billions(model_id)
    need_gb = size_b * (0.65 if quant_resolved == "4bit" else 2.1)
    warnings = []
    ram = hw.get("ram_available_gb")
    if hw["cuda"]:
        vram = hw.get("gpu_mem_gb") or 0
        if vram < need_gb:
            warnings.append(f"GPU memory ({vram} GB) is smaller than the ~{need_gb:.0f} GB the model needs; "
                            f"some layers will be placed in CPU RAM (slower).")
            if ram is not None and vram + ram < need_gb and not force:
                raise ModelResourceError(
                    f"Not enough memory: the model needs about {need_gb:.0f} GB but this machine has "
                    f"{vram} GB GPU + {ram} GB free RAM. Use 4-bit loading, a smaller model ID "
                    f"(e.g. meta-llama/Llama-3.2-3B-Instruct), or the heuristic demo engine. "
                    f"Tick 'Try anyway' to override.")
    else:
        if ram is not None and ram < need_gb:
            msg = (f"Not enough free RAM: the model needs about {need_gb:.0f} GB but only {ram} GB is free. "
                   f"Llama 3.1 8B needs a GPU (with 4-bit) or 16+ GB free RAM. Try a smaller model ID "
                   f"(e.g. meta-llama/Llama-3.2-1B-Instruct) or the heuristic demo engine.")
            if not force:
                raise ModelResourceError(msg + " Tick 'Try anyway' to override.")
            warnings.append(msg)
        warnings.append("No GPU detected: generation on CPU will be slow.")
    return warnings


def translate_error(exc, model_id):
    """Turn low-level exceptions into ModelError subclasses with plain-language advice."""
    if isinstance(exc, ModelError):
        return exc
    name, msg = type(exc).__name__, str(exc)
    low = msg.lower()
    if "gated" in low or name in ("GatedRepoError",) or "401" in msg or "403" in msg or \
            "access to model" in low or "restricted" in low:
        return ModelAccessError(
            f"Access to '{model_id}' was denied. Llama models are gated: (1) open the model page on "
            f"huggingface.co and accept Meta's license, (2) create a read token at "
            f"huggingface.co/settings/tokens, (3) log in with 'huggingface-cli login' or paste the token in "
            f"the sidebar / set the HF_TOKEN environment variable. Original error: {msg[:200]}")
    if name in ("RepositoryNotFoundError", "HFValidationError") or "not a valid model identifier" in low \
            or "is not a local folder" in low:
        return ModelDownloadError(f"Model '{model_id}' was not found. Check the spelling or the local folder path.")
    if "outofmemory" in name.lower() or "out of memory" in low or isinstance(exc, MemoryError):
        return ModelResourceError(
            "Out of memory while loading/running the model. Try 4-bit loading, a smaller model ID, "
            "closing other programs, or the heuristic demo engine.")
    if any(k in low for k in ("connection", "timed out", "network", "offline", "max retries", "resolve")):
        return ModelDownloadError(
            f"Could not download '{model_id}' (network problem). Check your internet connection and try again. "
            f"The download is large (about 16 GB for Llama 3.1 8B). Original error: {msg[:200]}")
    if "no space left" in low:
        return ModelDownloadError("Not enough free disk space to download the model.")
    return ModelError(f"Could not load the model: {name}: {msg[:300]}")


def load_base_model(model_id, quant="auto", token=None, force=False, for_training=False):
    """Load (model, tokenizer, notes). Raises ModelError subclasses with friendly messages."""
    require_packages(["torch", "transformers", "accelerate"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    notes = []
    try:
        cuda = torch.cuda.is_available()
        q, note = resolve_quantization(quant, cuda)
        if note:
            notes.append(note)
        notes += check_memory(model_id, q, force)
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = {"token": token, "low_cpu_mem_usage": True}
        if cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.bfloat16
        if q == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = "auto"
        elif cuda:
            kwargs["torch_dtype"] = dtype
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = dtype
        log.info("Loading %s (quantization=%s, cuda=%s)...", model_id, q, cuda)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not cuda:
            model.to("cpu")
        if not for_training:
            model.eval()
        notes.append(f"Loaded with quantization={q}, device={'cuda' if cuda else 'cpu'}.")
        return model, tok, notes
    except ModelError:
        raise
    except Exception as exc:  # translate anything else
        raise translate_error(exc, model_id) from exc


def adapter_info(adapter_dir):
    """Return metadata dict if a *really trained* adapter exists, else None."""
    adapter_dir = Path(adapter_dir)
    meta = load_json(adapter_dir / "training_meta.json")
    if (adapter_dir / "adapter_config.json").exists() and meta and meta.get("trained") is True:
        return meta
    return None


class LLMEngine:
    """Llama via Hugging Face Transformers. kind = 'pretrained' or 'finetuned' (LoRA adapter)."""
    is_llm = True

    def __init__(self, model_id=config.DEFAULT_MODEL_ID, adapter_dir=None, quant="auto",
                 token=None, force=False):
        self.model_id = model_id
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.kind = "finetuned" if self.adapter_dir else "pretrained"
        self.quant, self.token, self.force = quant, token or None, force
        self.model = self.tokenizer = None
        self.notes, self.adapter_meta, self.has_adapter = [], None, False

    @property
    def name(self):
        return f"{self.model_id} ({'LoRA fine-tuned' if self.kind == 'finetuned' else 'pretrained, few-shot'})"

    def load(self):
        if self.adapter_dir is not None:
            self.adapter_meta = adapter_info(self.adapter_dir)
            if self.adapter_meta is None:
                raise ModelError(
                    f"No trained LoRA adapter was found in '{self.adapter_dir}'. Fine-tuning has not been "
                    f"run yet. Run train_llama_sample.bat (or 'python train.py') first, or choose the "
                    f"pretrained mode.")
            require_packages(["peft"])
        self.model, self.tokenizer, self.notes = load_base_model(self.model_id, self.quant, self.token, self.force)
        if self.adapter_dir is not None:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, str(self.adapter_dir))
                self.model.eval()
                self.has_adapter = True
                base = self.adapter_meta.get("base_model_id")
                if base and base != self.model_id:
                    self.notes.append(f"Warning: adapter was trained on '{base}', but '{self.model_id}' is loaded.")
            except Exception as exc:
                raise translate_error(exc, str(self.adapter_dir)) from exc
        return self

    def generate(self, messages, max_new_tokens=128, temperature=0.0, use_adapter=True):
        """Generate a reply. use_adapter=False runs the base model even if an adapter is loaded."""
        if self.model is None:
            raise ModelError("The model is not loaded yet.")
        import torch
        tok = self.tokenizer
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            eos = [tok.eos_token_id]
            eot = tok.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id:
                eos.append(eot)
            gen = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id, "eos_token_id": eos}
            if temperature and temperature > 0:
                gen.update(do_sample=True, temperature=float(temperature), top_p=0.9)
            else:
                gen.update(do_sample=False)
            ctx = contextlib.nullcontext()
            if self.has_adapter and not use_adapter:
                ctx = self.model.disable_adapter()
            with torch.no_grad(), ctx:
                out = self.model.generate(**enc, **gen)
            return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as exc:
            raise translate_error(exc, self.model_id) from exc


class HeuristicEngine:
    """ML-backed engine using the trained project models."""
    is_llm = False
    kind = "ml"
    name = "Trained ML model"
    model_id = "tfidf_logreg"
    notes = ["Uses the trained TF-IDF + Logistic Regression model."]

    def __init__(self):
        from predict import MLPredictor
        self.predictor = MLPredictor()

    def predict_discourse_type(self, text):
        return self.predictor.predict_discourse_type(text)

    def predict_effectiveness(self, text):
        return self.predictor.predict_effectiveness(text)

    def analyze_essay(self, text):
        return self.predictor.analyze_essay(text)


def create_engine(mode="heuristic", model_id=config.DEFAULT_MODEL_ID, adapter_dir=None,
                  quant="auto", token=None, force=False):
    """mode: 'heuristic' | 'pretrained' | 'finetuned'. Returns a ready-to-use engine."""
    if mode == "heuristic":
        return HeuristicEngine()
    if mode == "pretrained":
        return LLMEngine(model_id, None, quant, token, force).load()
    if mode == "finetuned":
        return LLMEngine(model_id, adapter_dir or config.LLAMA_ADAPTER_DIR, quant, token, force).load()
    raise ValueError(f"Unknown engine mode: {mode}")
