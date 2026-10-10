"""Smoke tests that run WITHOUT downloading any model.  Run:  python tests/smoke_test.py

They cover data loading, segmentation, classification, assessment, the Socratic workflow,
evaluation and the highlighting HTML with the heuristic engine, and they exercise the LLM code
path (prompt parsing, JSON parsing, fallbacks) with a scripted STUB in place of Llama.
The stub is not a language model: these tests do NOT prove Llama itself works.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import evaluate
from data_loader import DataLoaderError, get_dataset, load_dataset_csv, split_dataset
from feedback import compare_assessments
from model import HeuristicEngine
from predict import ArgumentAnalyzer, select_weak_arguments
from preprocessing import parse_classification, segment_essay, validate_essay
from socratic_feedback import SocraticSession, clean_question
from utils import ensure_dirs, extract_json, render_highlighted_html

passed = []


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    passed.append(name)
    print("PASS", name)


class StubLLM:
    """Scripted stand-in for Llama: returns canned strings depending on the prompt."""
    is_llm, kind, name, model_id = True, "pretrained", "stub", "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, max_new_tokens=64, temperature=0.0, use_adapter=True):
        self.calls += 1
        system = messages[0]["content"]
        if "Classify one text span" in system:
            return "Type: Claim\nQuality: Adequate"
        if "argument-quality assessor" in system:
            return ('Sure! {"score": 3, "effectiveness": "Ineffective", "has_evidence": false, '
                    '"has_reasoning": true, "weakness_types": ["unclear_claim", "missing_evidence"], '
                    '"strengths": ["Clear position"], "weaknesses": ["Vague word easy"], '
                    '"explanation": "The word easy is unclear and no evidence is given."}')
        if "Socratic" in system:
            return "What do you mean by 'easy' and easier than what?"
        if "writing coach" in system:
            return '{"suggestions": ["Name one example.", "Explain the link.", "Define easy."]}'
        return ""


ensure_dirs()
essay = (Path(config.SAMPLE_ESSAYS_DIR) / "sample_essay.txt").read_text(encoding="utf-8")

# ---- data loading
df, rep = get_dataset("sample")
check("sample dataset loads (36 rows)", len(df) == 36 and rep["source"] == "sample")
try:
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.csv"
        bad.write_text("a,b\n1,2\n")
        load_dataset_csv(bad)
    check("missing columns rejected", False)
except DataLoaderError:
    check("missing columns rejected", True)
tr, va, te = split_dataset(df)
check("essay-grouped split has no overlap",
      not (set(tr.essay_id) & set(te.essay_id)) and len(tr) + len(va) + len(te) == len(df))
try:
    get_dataset("feedback-prize", Path(tempfile.mkdtemp()))
    check("empty dataset dir rejected", False)
except DataLoaderError:
    check("empty dataset dir rejected", True)

# ---- preprocessing
try:
    validate_essay("too short")
    check("short essay rejected", False)
except ValueError:
    check("short essay rejected", True)
segs = segment_essay(essay)
check("segmentation offsets match text", all(essay[s["start"]:s["end"]] == s["text"] for s in segs))
check("parse_classification", parse_classification("Type: rebuttal\nQuality: effective") == ("Rebuttal", "Effective"))
check("extract_json tolerant", extract_json('blah {"a": 1,} tail') == {"a": 1})

# ---- heuristic workflow
an = ArgumentAnalyzer(HeuristicEngine())
res = an.analyze_essay(essay)
types = {s["type"] for s in res["segments"]}
check("analysis returns typed segments", len(res["segments"]) >= 8 and "Evidence" in types)
check("summary counts consistent", sum(res["summary"]["type_counts"].values()) == len(res["segments"]))
html_out = render_highlighted_html(res["essay"], res["segments"])
check("highlight html has type tags", "Evidence" in html_out and "<script" not in html_out)
weak = select_weak_arguments(res)
check("weak arguments found in sample essay", len(weak) >= 1)
s = SocraticSession(an, res["essay"], weak[0])
q1 = s.start()
check("first question is a question", q1.strip().endswith("?"))
check("suggestions locked before attempt", not s.can_view_suggestions)
try:
    s.get_suggestions()
    check("locked suggestions raise", False)
except PermissionError:
    check("locked suggestions raise", True)
q2 = s.reply("Students can learn from home and save travelling time.")
check("follow-up generated", q2 != q1 and len(s.history) == 3)
q3 = s.reply("Because they save two hours a day, they can study more, which means better grades.")
check("multi-round history kept", len(s.history) == 5 and s.rounds == 2)
rev = s.revise("Online education helps working adults because a 2020 survey found 60 percent save an hour of travel daily.")
check("revision reassessed with comparison", "delta" in rev["comparison"] and s.revisions)
check("suggestions unlocked after attempt", len(s.get_suggestions()["suggestions"]) >= 1)
try:
    s.reply("   ")
    check("empty answer rejected", False)
except ValueError:
    check("empty answer rejected", True)
card = s.feedback_card()
check("feedback card complete", card["exchanges"] and card["identified_weakness"] and card["original_argument"])
check("session saves to json", json.loads(Path(s.save()).read_text(encoding="utf-8"))["card"])
check("compare_assessments", compare_assessments({"score": 3, "weakness_types": ["a"]},
                                                 {"score": 5, "weakness_types": []})["delta"] == 2)

# ---- LLM code path with stub
stub = ArgumentAnalyzer(StubLLM())
r2 = stub.analyze_essay(essay)
check("stub: LLM classification parsed", all(x["source"] == "llm-fewshot" for x in r2["segments"] if x["type"] != "Other"))
a0 = next(x for x in r2["segments"] if x.get("assessment"))["assessment"]
check("stub: LLM JSON assessment parsed", a0["source"] == "llm" and a0["score"] == 3)
w2 = select_weak_arguments(r2)
s2 = SocraticSession(stub, r2["essay"], w2[0])
check("stub: LLM Socratic question used", s2.start().endswith("?") and s2.history[0]["source"] == "llm")
check("stub: LLM follow-up", s2.reply("Because it saves time.").endswith("?"))
check("stub: LLM suggestions", stub and SocraticSession.get_suggestions.__name__ and s2.can_view_suggestions
      and s2.get_suggestions()["source"] == "llm")
check("clean_question rejects non-questions", clean_question("Here is the answer.") is None)


class BadStub(StubLLM):
    def generate(self, messages, **kw):
        return "I cannot help"


bad = ArgumentAnalyzer(BadStub())
b_as = bad.assess_argument("Online education is better because it is easy.", "Position", essay)
check("bad LLM output falls back to rules (marked)", b_as["source"] == "heuristic-fallback")
b_s = SocraticSession(bad, essay, {"idx": 0, "type": "Position", "text": "Online education is easy.",
                                   "assessment": b_as})
check("bad LLM question falls back to template (marked)", b_s.start().endswith("?") and
      b_s.history[0]["source"] == "template-fallback")

# ---- evaluation
with tempfile.TemporaryDirectory() as d:
    out = evaluate.evaluate_predictor(an, df, "heuristic_test", "sample", "all", output_root=Path(d))
    files = {p.name for p in Path(out["output_dir"]).iterdir()}
    check("evaluation writes report files", {"metrics.json", "predictions.csv", "confusion_type.png", "report.txt"} <= files)
    check("metrics in [0,1]", 0 <= out["type"]["accuracy"] <= 1 and 0 <= out["effectiveness"]["f1_macro"] <= 1)
    tbl = evaluate.comparison_table(evaluate.load_runs(Path(d)))
    check("comparison table built from saved runs", len(tbl) == 1)

print(f"\nALL {len(passed)} CHECKS PASSED")
