"""Shared helpers: folders, logging, JSON, text utilities, hardware info, HTML highlighting."""
import hashlib
import html
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import config


def ensure_dirs():
    """Create every folder the project needs."""
    for d in config.ALL_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_logger(name="argmine"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def extract_json(text):
    """Extract the first JSON object from model output. Returns dict or None."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start:end + 1]
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate),
                    re.sub(r",\s*([}\]])", r"\1", candidate).replace("'", '"')):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def word_count(text):
    return len(re.findall(r"\w+", text or ""))


def truncate_words(text, n):
    words = (text or "").split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def short_quote(text, n=14):
    words = (text or "").strip().split()
    return " ".join(words) if len(words) <= n else " ".join(words[:n]) + " ..."


def decode_upload(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file as text.")


def fingerprint(items):
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def hardware_summary():
    info = {"torch": None, "cuda": False, "gpu_name": None, "gpu_mem_gb": None,
            "ram_total_gb": None, "ram_available_gb": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_mem_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except ImportError:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024 ** 3, 1)
        info["ram_available_gb"] = round(vm.available / 1024 ** 3, 1)
    except ImportError:
        pass
    return info


def _esc_with_cues(text, show_reasoning):
    """HTML-escape text, wrapping rule-based reasoning cues (because, therefore...) if requested."""
    if not show_reasoning:
        return html.escape(text)
    out, pos = [], 0
    for m in re.finditer(config.REASONING_CUES, text, flags=re.IGNORECASE):
        out.append(html.escape(text[pos:m.start()]))
        out.append(f'<span style="background:{config.REASONING_CUE_COLOR};color:#111;'
                   f'border-radius:3px;padding:0 2px;" title="reasoning cue (rule-based)">'
                   f'{html.escape(m.group(0))}</span>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def render_highlighted_html(text, segments, show_reasoning=True):
    """Return HTML of the essay with argument spans colored by predicted type."""
    parts, pos = [], 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["start"] < pos:
            continue
        parts.append(html.escape(text[pos:seg["start"]]))
        seg_text = text[seg["start"]:seg["end"]]
        label = seg.get("type", "Other")
        if label == "Other" or label not in config.TYPE_COLORS:
            parts.append(html.escape(seg_text))
        else:
            color = config.TYPE_COLORS[label]
            parts.append(
                f'<span style="background:{color};color:#111;padding:2px 3px;border-radius:4px;">'
                f'{_esc_with_cues(seg_text, show_reasoning)}'
                f'<span style="font-size:0.68em;font-weight:700;background:#334155;color:#fff;'
                f'border-radius:3px;padding:0 4px;margin-left:4px;vertical-align:middle;">'
                f'{html.escape(label)}</span></span>')
        pos = seg["end"]
    parts.append(html.escape(text[pos:]))
    return ('<div style="white-space:pre-wrap;line-height:2.0;font-size:1.02em;">'
            + "".join(parts) + "</div>")


def legend_html():
    items = [f'<span style="background:{c};color:#111;padding:2px 8px;border-radius:4px;'
             f'margin-right:6px;">{t}</span>'
             for t, c in config.TYPE_COLORS.items() if t != "Other"]
    items.append(f'<span style="background:{config.REASONING_CUE_COLOR};color:#111;padding:2px 8px;'
                 f'border-radius:4px;">reasoning cue (rule-based)</span>')
    return "<div style='line-height:2.2'>" + "".join(items) + "</div>"
