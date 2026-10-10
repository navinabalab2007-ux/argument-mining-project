"""Improvement suggestions (shown only AFTER the student has attempted to improve) and score comparison."""
import config
import heuristic_engine as heur
from predict import WEAKNESS_LABELS
from utils import extract_json, truncate_words


def generate_suggestions(analyzer, essay_text, arg_text, arg_type, assessment):
    """Return {'suggestions': [...], 'source': 'llm'|'heuristic'|'heuristic-fallback'}.

    Suggestions are hints for the student. They never contain a rewritten argument.
    """
    if not analyzer.is_llm:
        return {"suggestions": heur.suggestions(arg_text, assessment), "source": "heuristic"}
    weaknesses = "; ".join(assessment.get("weaknesses", [])) or "none identified"
    msgs = [{"role": "system", "content": _suggest_system()},
            {"role": "user", "content": f"Essay context:\n\"\"\"{truncate_words(essay_text, 250)}\"\"\"\n\n"
                                        f"Argument type: {arg_type}\nArgument:\n\"\"\"{arg_text}\"\"\"\n"
                                        f"Identified weaknesses: {weaknesses}\nReturn the JSON object."}]
    raw = analyzer.generate(msgs, config.SUGGEST_MAX_NEW_TOKENS, 0.3, use_adapter=False)
    obj = extract_json(raw)
    items = obj.get("suggestions") if obj else None
    if isinstance(items, list):
        items = [str(i).strip() for i in items if str(i).strip()][:5]
        if items:
            return {"suggestions": items, "source": "llm"}
    return {"suggestions": heur.suggestions(arg_text, assessment), "source": "heuristic-fallback"}


def _suggest_system():
    from predict import SUGGEST_SYSTEM
    return SUGGEST_SYSTEM


def compare_assessments(old, new):
    """Compare two assessments of the same argument (before/after revision)."""
    old_w, new_w = set(old["weakness_types"]), set(new["weakness_types"])
    delta = new["score"] - old["score"]
    if delta > 0:
        msg = f"The score went up by {delta} point(s). Your revision strengthened the argument."
    elif delta == 0:
        msg = "The score stayed the same. Keep going: look at the weaknesses that remain."
    else:
        msg = f"The score went down by {abs(delta)} point(s). Compare the two versions and think about what changed."
    return {"old_score": old["score"], "new_score": new["score"], "delta": delta,
            "resolved": sorted(old_w - new_w), "remaining": sorted(old_w & new_w),
            "new_issues": sorted(new_w - old_w), "message": msg,
            "resolved_labels": [WEAKNESS_LABELS.get(w, w) for w in sorted(old_w - new_w)],
            "remaining_labels": [WEAKNESS_LABELS.get(w, w) for w in sorted(old_w & new_w)]}
