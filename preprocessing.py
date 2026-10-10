"""Essay segmentation, prompt construction and output parsing."""
import re

import config
from utils import truncate_words, word_count

# ------------------------------------------------------------------ validation / segmentation


def clean_essay(text):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def validate_essay(text):
    """Raise ValueError with a friendly message if the essay cannot be analysed."""
    if not text or not text.strip():
        raise ValueError("The essay is empty. Paste an essay or upload a .txt file.")
    n = word_count(text)
    if n < config.MIN_ESSAY_WORDS:
        raise ValueError(f"The essay is too short ({n} words). Please provide at least "
                         f"{config.MIN_ESSAY_WORDS} words.")
    if n > config.MAX_ESSAY_WORDS:
        raise ValueError(f"The essay is too long ({n} words). The limit is {config.MAX_ESSAY_WORDS}.")
    return True


_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"')\]]*|$)")


def segment_essay(text):
    """Split an essay into sentence-level segments with character offsets.

    Each segment: idx, text, start, end, paragraph_idx, para_count, sent_idx_in_para.
    Every non-empty line is treated as a paragraph.
    """
    paragraphs = [m for m in re.finditer(r"[^\n]+", text) if m.group(0).strip()]
    segments = []
    for p_idx, pm in enumerate(paragraphs):
        para, base = pm.group(0), pm.start()
        s_in_p = 0
        for sm in _SENT_RE.finditer(para):
            raw = sm.group(0)
            stripped = raw.strip()
            if not re.search(r"\w", stripped):
                continue
            lead = len(raw) - len(raw.lstrip())
            start = base + sm.start() + lead
            end = start + len(stripped)
            segments.append({"idx": len(segments), "text": stripped, "start": start, "end": end,
                             "paragraph_idx": p_idx, "para_count": len(paragraphs),
                             "sent_idx_in_para": s_in_p})
            s_in_p += 1
    total = len(segments)
    for s in segments:
        s["essay_frac"] = s["idx"] / max(1, total)
    return segments


# ------------------------------------------------------------------ prompts

TYPE_DEFINITIONS = """Argument element types:
- Lead: an introduction that grabs attention (statistic, quotation, question, description).
- Position: the writer's opinion or answer to the main question.
- Claim: a statement that supports the position.
- Counterclaim: a statement that opposes another claim or the position.
- Rebuttal: a statement that refutes a counterclaim.
- Evidence: facts, examples or data that support a claim, counterclaim or rebuttal.
- Concluding Statement: a closing statement that restates the position or claims.
- Other: text that is not part of an argument."""

CLASSIFY_SYSTEM = (
    "You analyse student essays. Classify one text span from an essay.\n" + TYPE_DEFINITIONS +
    "\nAlso rate how effective the span is: Ineffective, Adequate or Effective.\n"
    "Answer with exactly two lines:\nType: <one type>\nQuality: <Ineffective|Adequate|Effective>")

# Hand-written illustrative examples (NOT taken from the Feedback Prize dataset).
FEW_SHOT_EXAMPLES = [
    ("Have you ever wondered why so many students feel tired in the morning?", "Lead", "Adequate"),
    ("I believe that schools should start later in the day.", "Position", "Adequate"),
    ("Later start times give teenagers time to sleep, which helps them concentrate in class.", "Claim", "Effective"),
    ("For example, a district in Minnesota reported higher attendance after it moved the bell by one hour.", "Evidence", "Effective"),
    ("Some people say that later starts would disrupt sports schedules.", "Counterclaim", "Adequate"),
    ("However, practices could be moved to the evening without losing any time.", "Rebuttal", "Adequate"),
    ("In conclusion, a later start would improve learning.", "Concluding Statement", "Adequate"),
    ("Thanks for reading my essay.", "Other", None),
]


def _query(text):
    text = truncate_words(text.strip(), config.MAX_SEGMENT_WORDS_FOR_MODEL)
    return f'Essay span:\n"""{text}"""\nClassify this span.'


def format_target(disc_type, effectiveness):
    return f"Type: {disc_type}\nQuality: {effectiveness or config.NOT_APPLICABLE}"


def build_classification_messages(text, few_shot=True):
    """Chat messages for classification. few_shot=True adds example turns (pretrained mode);
    few_shot=False is the zero-shot format used to fine-tune the model."""
    msgs = [{"role": "system", "content": CLASSIFY_SYSTEM}]
    if few_shot:
        for ex_text, ex_type, ex_eff in FEW_SHOT_EXAMPLES:
            msgs.append({"role": "user", "content": _query(ex_text)})
            msgs.append({"role": "assistant", "content": format_target(ex_type, ex_eff)})
    msgs.append({"role": "user", "content": _query(text)})
    return msgs


def match_label(value, labels):
    """Fuzzy-match a string to one of the labels. Returns label or None."""
    if not value:
        return None
    v = re.sub(r"[^a-z/ ]", "", value.lower()).strip()
    if not v:
        return None
    for lab in labels:
        if v == lab.lower():
            return lab
    for lab in sorted(labels, key=len, reverse=True):
        if lab.lower() in v:
            return lab
    first = v.split()[0]
    for lab in labels:
        if lab.lower().startswith(first) and len(first) >= 4:
            return lab
    return None


def parse_classification(output):
    """Parse 'Type: X / Quality: Y' model output -> (type or None, effectiveness or None)."""
    if not output:
        return None, None
    t = re.search(r"type\s*[:\-]\s*([^\n]+)", output, re.IGNORECASE)
    q = re.search(r"quality\s*[:\-]\s*([^\n]+)", output, re.IGNORECASE)
    dtype = match_label(t.group(1), config.SEGMENT_TYPES) if t else match_label(output, config.SEGMENT_TYPES)
    eff = match_label(q.group(1), config.EFFECTIVENESS_LABELS) if q else None
    return dtype, eff
