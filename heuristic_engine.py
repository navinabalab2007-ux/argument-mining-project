"""Rule-based DEMO engine. It runs instantly on any laptop without downloading a model.

IMPORTANT: this is NOT Llama and NOT a trained model. It uses simple word-pattern rules so the
whole application (highlighting, dashboard, Socratic chat) can be tried without a GPU. Results
from it are always labelled 'heuristic' in the interface.
"""
import re

import config
from utils import short_quote

CONCLUDE = re.compile(r"\b(in conclusion|to conclude|in summary|to sum up|all in all|overall|"
                      r"in the end|finally,)\b", re.I)
COUNTER = re.compile(r"\b(some (people|argue|say|believe|parents|students)|others (argue|say|believe)|"
                     r"critics|opponents|it (is|may be) (argued|said)|one might (argue|say)|"
                     r"(might|may) (say|argue|worry|think))\b", re.I)
REBUT = re.compile(r"^(however|but|yet|nevertheless|on the other hand|still|even so|although)\b", re.I)
EVID = re.compile(r"(\bfor (example|instance)\b|\bsuch as\b|\baccording to\b|\bstud(y|ies)\b|"
                  r"\bresearch\b|\bsurvey\b|\bstatistics?\b|\bpercent\b|%|\b(19|20)\d{2}\b|\d)", re.I)
POSITION = re.compile(r"\b(i (believe|think|feel|agree|disagree|strongly)|in my (opinion|view)|"
                      r"should|must|is (bad|good|better|worse)|are (bad|good|better|worse))\b", re.I)
LEAD_MARK = re.compile(r"(\?$|\bimagine\b|\bhave you ever\b|\bwhat if\b)", re.I)
CLAIMWORDS = re.compile(r"\b(should|must|better|worse|important|beneficial|harmful|because|therefore|"
                        r"cause[sd]?|leads?|helps?|improv\w+|allow\w*|enable\w*|need|reduce\w*|"
                        r"increase\w*|can|will|would|could|too)\b", re.I)
REASON = re.compile(config.REASONING_CUES, re.I)
VAGUE = {"easy", "easier", "good", "bad", "nice", "great", "very", "really", "stuff", "thing",
         "things", "lot", "everyone", "everybody", "always", "never", "better", "best", "worse",
         "important", "many", "some", "hard"}


def _features(text):
    tokens = re.findall(r"[A-Za-z0-9'%]+", text)
    low = [t.lower() for t in tokens]
    proper = sum(1 for i, w in enumerate(tokens) if i > 0 and w[0].isupper() and w != "I")
    specifics = len(EVID.findall(text)) + proper
    return {"n": len(tokens), "specifics": specifics,
            "reasoning": len(REASON.findall(text)),
            "vague": sum(1 for w in low if w in VAGUE),
            "vague_words": [w for w in dict.fromkeys(low) if w in VAGUE]}


def _effectiveness(score):
    return "Effective" if score >= 7 else "Adequate" if score >= 4 else "Ineffective"


def assess(text, arg_type="Claim"):
    """Rule-based quality estimate. Returns the same dict shape as the LLM assessment."""
    f = _features(text)
    score = 4.0
    strengths, weaknesses, wtypes = [], [], []
    if f["n"] >= 12:
        score += 1
    if f["n"] >= 20:
        score += 0.5
    if arg_type == "Evidence":
        if f["specifics"]:
            score += 2.5
            strengths.append("It includes concrete details (numbers, examples, or named sources).")
        else:
            score -= 1
            wtypes.append("too_vague")
            weaknesses.append("The evidence has no concrete detail such as a number, name, or example.")
        if f["reasoning"]:
            score += 0.5
    else:
        if f["specifics"]:
            score += 2
            strengths.append("It points to specific details or examples.")
        elif arg_type in config.CLAIM_GROUP:
            wtypes.append("missing_evidence")
            weaknesses.append("No example, fact, or source supports this statement.")
        if f["reasoning"]:
            score += 1.5
            strengths.append("It explains a reason (a connecting word such as 'because').")
        elif arg_type in config.CLAIM_GROUP and f["n"] >= 5:
            wtypes.append("weak_reasoning")
            weaknesses.append("The link between the idea and its conclusion is not explained.")
    score -= min(2.0, 0.75 * f["vague"])
    if f["vague"] and f["n"] < 25:
        wtypes.insert(0, "unclear_claim")
        weaknesses.insert(0, "Some wording is vague (" + ", ".join(f["vague_words"][:3]) +
                          "), so the exact meaning is unclear.")
    if f["n"] < 8:
        score -= 1
        if "unclear_claim" not in wtypes:
            wtypes.insert(0, "unclear_claim")
            weaknesses.insert(0, "The statement is very short and leaves the idea underdeveloped.")
    score = max(1, min(10, int(score + 0.5)))
    if f["n"] >= 12 and not weaknesses:
        strengths.append("It is developed in enough words to be understood.")
    expl = (f"Rule-based estimate (not a model judgement): {f['n']} words, "
            f"{f['specifics']} concrete-detail cue(s), {f['reasoning']} reasoning cue(s), "
            f"{f['vague']} vague word(s).")
    return {"score": score, "effectiveness": _effectiveness(score),
            "has_evidence": f["specifics"] > 0, "has_reasoning": f["reasoning"] > 0,
            "weakness_types": list(dict.fromkeys(wtypes)), "strengths": strengths,
            "weaknesses": weaknesses, "explanation": expl, "source": "heuristic"}


def classify(text, ctx=None):
    """Rule-based span classification. ctx (optional) holds position info from segmentation."""
    ctx = ctx or {}
    low = text.lower()
    n = len(re.findall(r"\w+", text))
    if n < config.MIN_SEGMENT_WORDS:
        return {"type": "Other", "effectiveness": config.NOT_APPLICABLE}
    multi = ctx.get("para_count", 1) >= 3
    intro = ctx.get("paragraph_idx", None) == 0 if ctx.get("para_count", 1) > 1 else ctx.get("essay_frac", 0.5) < 0.2
    last = (ctx.get("paragraph_idx", -1) == ctx.get("para_count", 0) - 1) if multi else \
        (ctx.get("essay_frac", 0.0) > 0.85 and ctx.get("para_count", 1) == 1)
    has_ev = bool(EVID.search(text))
    if CONCLUDE.search(low) or (last and not has_ev and multi):
        label = "Concluding Statement"
    elif COUNTER.search(low):
        label = "Counterclaim"
    elif REBUT.match(low) and ctx.get("prev_type") == "Counterclaim":
        label = "Rebuttal"
    elif intro and POSITION.search(low):
        label = "Position"
    elif has_ev:
        label = "Evidence"
    elif intro and ctx.get("idx", 1) == 0 and (LEAD_MARK.search(text) or not CLAIMWORDS.search(low)):
        label = "Lead"
    elif CLAIMWORDS.search(low):
        label = "Claim"
    else:
        label = "Other"
    if label in ("Other",):
        return {"type": label, "effectiveness": config.NOT_APPLICABLE}
    return {"type": label, "effectiveness": _effectiveness(assess(text, label)["score"])}


# --------------------------------------------------------------------------- Socratic templates

_FIRST = {
    "unclear_claim": [
        "You wrote: \"{q}\" Which words in that sentence could mean different things to different "
        "readers, and what exactly do you mean by them?",
        "When you say \"{q}\", who is it true for, and compared with what?"],
    "missing_evidence": [
        "You wrote: \"{q}\" What real example, fact, or experience could you point to that shows this is true?",
        "If a classmate doubted \"{q}\", what could you show them to convince them?"],
    "weak_reasoning": [
        "You wrote: \"{q}\" What steps connect that idea to your conclusion? Try to walk me through them.",
        "How does the first part of \"{q}\" lead to the result you describe?"],
    "too_vague": [
        "You mention evidence: \"{q}\" What specific detail (who, when, how many) would make it easier to trust?"],
    "off_topic": [
        "How does \"{q}\" connect back to the main question of your essay?"],
    "deepen": [
        "This argument is already fairly developed: \"{q}\" What is the strongest objection someone could raise, and how would you respond?"],
}


def first_question(text, weakness):
    pool = _FIRST.get(weakness, _FIRST["deepen"])
    return pool[0].format(q=short_quote(text, 16))


def followup_question(argument_text, weakness, student_answer, round_no):
    """Template follow-up based on the features of the student's answer (no answer is revealed)."""
    ans = student_answer.strip()
    f = _features(ans)
    aq = short_quote(ans, 12).rstrip(" .!?")
    claim = short_quote(argument_text, 12)
    if f["n"] < 6:
        return (f"Can you say more? You answered \"{aq}\". Add one detail, so a reader who "
                f"disagrees with you could still follow your thinking.")
    if f["specifics"] and not f["reasoning"]:
        return (f"You gave a detail: \"{aq}\". How does that detail show that \"{claim}\" is true?")
    if f["reasoning"] and not f["specifics"]:
        return (f"You explained a reason: \"{aq}\". What real example or piece of evidence could "
                f"back up that reason?")
    if f["reasoning"] and f["specifics"]:
        return ("You now have both a reason and a concrete detail. Try rewriting your original argument "
                "using them. What would your new version say?")
    if weakness == "unclear_claim":
        return (f"You said \"{aq}\". Can you make that more precise, for instance by saying who "
                f"benefits, how much, or in what situation?")
    return (f"You said \"{aq}\". What would a reader who disagrees say about that, and how would you respond?")


_SUGGEST = {
    "unclear_claim": "Replace vague words with precise ones. Say exactly who, what, and compared with what.",
    "missing_evidence": "Add one concrete example, statistic, or source that supports the statement, and introduce it with a phrase like 'for example'.",
    "weak_reasoning": "Add a sentence that explains how your point leads to your conclusion (use 'because', 'which means', or 'as a result').",
    "too_vague": "Make the evidence specific: name the source, give a number, or describe one real situation.",
    "off_topic": "Connect this sentence back to your main position in a closing clause.",
}


def suggestions(text, assessment):
    out = [_SUGGEST[w] for w in assessment.get("weakness_types", []) if w in _SUGGEST]
    if not out:
        out = ["This argument is fairly strong. Consider addressing a possible objection to make it more convincing.",
               "Check that each sentence adds new information rather than repeating the point."]
    out.append("Read your revised sentence aloud: could a skeptical reader still ask 'why?' If so, answer that question in writing.")
    return out
