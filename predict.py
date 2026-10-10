"""
ML-based Argument Analyzer

Effectiveness:
    TF-IDF + Logistic Regression

Discourse Type:
    Auxiliary TF-IDF Classifier
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np


# ============================================================
# WEAKNESS LABELS
# ============================================================

WEAKNESS_LABELS = {
    "insufficient_evidence": "Insufficient evidence",
    "weak_reasoning": "Weak reasoning",
    "unclear_claim": "Unclear claim",
    "unsupported_claim": "Unsupported claim",
    "missing_counterargument": "Missing counterargument",
}


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EFFECTIVENESS_MODEL_DIR = (
    BASE_DIR / "models" / "saved" / "tfidf_logreg"
)

DISCOURSE_MODEL_DIR = (
    BASE_DIR / "models" / "saved" / "aux_discourse_type"
)

LABEL_ENCODER_FILE = (
    BASE_DIR / "data" / "processed" / "label_encoder.json"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text):

    return len(clean_text(text).split())


# ============================================================
# SENTENCE SEGMENTATION
# ============================================================

def segment_text(text):

    text = clean_text(text)

    if not text:
        return []

    segments = []

    pattern = r"[^.!?]+[.!?]?"

    for idx, match in enumerate(
        re.finditer(pattern, text)
    ):

        sentence = match.group(0).strip()

        if not sentence:
            continue

        start = match.start()
        end = match.end()

        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        segments.append(
            {
                "idx": idx,
                "text": text[start:end],
                "start": start,
                "end": end,
            }
        )

    return segments


# ============================================================
# ML PREDICTOR
# ============================================================

class MLPredictor:

    def __init__(self):

        self.effectiveness_model = None
        self.effectiveness_vectorizer = None

        self.discourse_model = None
        self.discourse_vectorizer = None

        self.effectiveness_labels = []

        self.discourse_labels = []

        self.load_models()

    # ========================================================
    # LOAD MODELS
    # ========================================================

    def load_models(self):

        # ----------------------------------------------------
        # Effectiveness model
        # ----------------------------------------------------

        effectiveness_vectorizer_path = (
            EFFECTIVENESS_MODEL_DIR
            / "vectorizer.joblib"
        )

        effectiveness_model_path = (
            EFFECTIVENESS_MODEL_DIR
            / "model.joblib"
        )

        if not effectiveness_vectorizer_path.exists():

            raise FileNotFoundError(
                "Effectiveness vectorizer not found:\n"
                f"{effectiveness_vectorizer_path}"
            )

        if not effectiveness_model_path.exists():

            raise FileNotFoundError(
                "Effectiveness model not found:\n"
                f"{effectiveness_model_path}"
            )

        self.effectiveness_vectorizer = (
            joblib.load(
                effectiveness_vectorizer_path
            )
        )

        self.effectiveness_model = (
            joblib.load(
                effectiveness_model_path
            )
        )

        # ----------------------------------------------------
        # Load original effectiveness labels
        # ----------------------------------------------------

        if LABEL_ENCODER_FILE.exists():

            with open(
                LABEL_ENCODER_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                encoder_data = json.load(f)

            if isinstance(encoder_data, dict):

                if "classes" in encoder_data:

                    self.effectiveness_labels = (
                        encoder_data["classes"]
                    )

                elif "labels" in encoder_data:

                    self.effectiveness_labels = (
                        encoder_data["labels"]
                    )

        # Fallback based on the training label order
        if not self.effectiveness_labels:

            self.effectiveness_labels = [
                "Adequate",
                "Effective",
                "Ineffective",
            ]

        # ----------------------------------------------------
        # Discourse model
        # ----------------------------------------------------

        discourse_vectorizer_path = (
            DISCOURSE_MODEL_DIR
            / "vectorizer.joblib"
        )

        discourse_model_path = (
            DISCOURSE_MODEL_DIR
            / "model.joblib"
        )

        if not discourse_vectorizer_path.exists():

            raise FileNotFoundError(
                "Discourse vectorizer not found:\n"
                f"{discourse_vectorizer_path}"
            )

        if not discourse_model_path.exists():

            raise FileNotFoundError(
                "Discourse model not found:\n"
                f"{discourse_model_path}"
            )

        self.discourse_vectorizer = (
            joblib.load(
                discourse_vectorizer_path
            )
        )

        self.discourse_model = (
            joblib.load(
                discourse_model_path
            )
        )

        # ----------------------------------------------------
        # Discourse labels
        # ----------------------------------------------------

        labels_file = (
            DISCOURSE_MODEL_DIR / "labels.json"
        )

        if labels_file.exists():

            with open(
                labels_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            self.discourse_labels = data.get(
                "labels",
                []
            )

    # ========================================================
    # EFFECTIVENESS PREDICTION
    # ========================================================

    def predict_effectiveness(self, text):

        text = clean_text(text)

        if not text:

            return (
                "Adequate",
                0.0,
                None
            )

        X = (
            self.effectiveness_vectorizer
            .transform([text])
        )

        prediction = (
            self.effectiveness_model
            .predict(X)[0]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Convert encoded number to original label
        # ----------------------------------------------------

        try:

            prediction_number = int(prediction)

            if (
                0 <= prediction_number
                < len(self.effectiveness_labels)
            ):

                prediction = (
                    self.effectiveness_labels[
                        prediction_number
                    ]
                )

        except (ValueError, TypeError):

            prediction = str(prediction)

        probabilities = None
        confidence = None

        if hasattr(
            self.effectiveness_model,
            "predict_proba"
        ):

            probabilities = (
                self.effectiveness_model
                .predict_proba(X)[0]
            )

            confidence = float(
                np.max(probabilities)
            )

        return (
            str(prediction),
            confidence,
            probabilities
        )

    # ========================================================
    # DISCOURSE TYPE
    # ========================================================

    def predict_discourse_type(self, text):

        text = clean_text(text)

        if not text:

            return "Other"

        X = (
            self.discourse_vectorizer
            .transform([text])
        )

        prediction = (
            self.discourse_model
            .predict(X)[0]
        )

        return str(prediction)


# ============================================================
# ARGUMENT ANALYZER
# ============================================================

class ArgumentAnalyzer:

    def __init__(self, predictor=None):

        if predictor is None:

            predictor = MLPredictor()

        self.predictor = predictor

        self.is_llm = False

        self.kind = "ml"

        self.name = (
            "TF-IDF + Logistic Regression + "
            "Auxiliary Discourse Classifier"
        )

    # ========================================================
    # CLASSIFY ONE SEGMENT
    # ========================================================

    def classify_segment(
        self,
        text,
        ctx=None
    ):

        text = clean_text(text)

        if word_count(text) < 4:

            return {
                "type": "Other",
                "effectiveness": "Adequate",
                "confidence": 0.0,
                "probabilities": None,
                "source": "trained-ml",
                "raw_output": "",
            }

        discourse_type = (
            self.predictor
            .predict_discourse_type(text)
        )

        (
            effectiveness,
            confidence,
            probabilities
        ) = (
            self.predictor
            .predict_effectiveness(text)
        )

        return {
            "type": str(discourse_type),
            "effectiveness": str(effectiveness),
            "confidence": confidence,
            "probabilities": probabilities,
            "source": "trained-ml",
            "raw_output": "",
        }

    # ========================================================
    # CLASSIFY MULTIPLE TEXTS
    # ========================================================

    def classify_texts(
        self,
        texts,
        progress_cb=None
    ):

        results = []

        total = len(texts)

        for i, text in enumerate(texts):

            result = self.classify_segment(text)

            results.append(
                {
                    "type": result["type"],
                    "effectiveness": result[
                        "effectiveness"
                    ],
                    "confidence": result[
                        "confidence"
                    ],
                }
            )

            if progress_cb:

                progress_cb(
                    (i + 1) / max(total, 1),
                    f"Classified {i + 1}/{total}"
                )

        return results

    # ========================================================
    # ARGUMENT QUALITY
    # ========================================================

    def assess_argument(
        self,
        arg_text,
        arg_type="Claim",
        essay_text=""
    ):

        (
            effectiveness,
            confidence,
            probabilities
        ) = (
            self.predictor
            .predict_effectiveness(arg_text)
        )

        # ----------------------------------------------------
        # Strengths / weaknesses
        # ----------------------------------------------------

        strengths = []
        weaknesses = []

        if effectiveness == "Effective":

            strengths.append(
                "The model classified this "
                "argument as Effective."
            )

        elif effectiveness == "Adequate":

            strengths.append(
                "The argument was classified "
                "as Adequate."
            )

            weaknesses.append(
                "The argument could be strengthened "
                "with clearer evidence or reasoning."
            )

        elif effectiveness == "Ineffective":

            weaknesses.append(
                "The model classified this "
                "argument as Ineffective."
            )

            weaknesses.append(
                "The argument needs stronger "
                "supporting reasoning or evidence."
            )

        # ----------------------------------------------------
        # Reasoning detection
        # ----------------------------------------------------

        has_reasoning = bool(
            re.search(
                r"\b("
                r"because|therefore|thus|since|"
                r"so|as a result|hence"
                r")\b",
                arg_text,
                re.IGNORECASE
            )
        )

        # ----------------------------------------------------
        # Evidence detection
        # ----------------------------------------------------

        has_evidence = bool(
            re.search(
                r"\b("
                r"study|research|survey|data|"
                r"statistics|according to|report|"
                r"example|evidence"
                r")\b",
                arg_text,
                re.IGNORECASE
            )
        )

        # ----------------------------------------------------
        # Weakness types
        # ----------------------------------------------------

        weakness_types = []

        if effectiveness == "Ineffective":

            weakness_types.append(
                "weak_reasoning"
            )

        if (
            not has_evidence
            and arg_type not in [
                "Evidence",
                "Lead",
                "Concluding Statement"
            ]
        ):

            weakness_types.append(
                "insufficient_evidence"
            )

        if (
            arg_type in [
                "Claim",
                "Position"
            ]
            and not has_reasoning
        ):

            weakness_types.append(
                "unclear_claim"
            )

        # ----------------------------------------------------
        # UI score
        # ----------------------------------------------------

        if effectiveness == "Effective":

            derived_score = 8

        elif effectiveness == "Adequate":

            derived_score = 6

        elif effectiveness == "Ineffective":

            derived_score = 3

        else:

            derived_score = 5

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        if confidence is not None:

            explanation = (
                f"The trained ML model classified "
                f"this segment as {effectiveness}. "
                f"Model confidence: "
                f"{confidence * 100:.1f}%."
            )

        else:

            explanation = (
                f"The trained ML model classified "
                f"this segment as {effectiveness}."
            )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {

            "score": derived_score,

            "effectiveness": effectiveness,

            "confidence": confidence,

            "has_evidence": has_evidence,

            "has_reasoning": has_reasoning,

            "weakness_types": weakness_types,

            "strengths": strengths,

            "weaknesses": weaknesses,

            "explanation": explanation,

            "source": "trained-ml",
        }

    # ========================================================
    # COMPLETE ESSAY ANALYSIS
    # ========================================================

    def analyze_essay(
        self,
        essay_text,
        progress_cb=None,
        assess=True
    ):

        essay_text = clean_text(
            essay_text
        )

        if word_count(essay_text) < 20:

            raise ValueError(
                "Essay must contain at least 20 words."
            )

        segments = segment_text(
            essay_text
        )

        if not segments:

            raise ValueError(
                "No sentences could be found "
                "in the essay."
            )

        total = len(segments)

        for i, segment in enumerate(
            segments
        ):

            result = self.classify_segment(
                segment["text"]
            )

            segment.update(result)

            if assess:

                segment["assessment"] = (
                    self.assess_argument(
                        segment["text"],
                        segment["type"],
                        essay_text
                    )
                )

            else:

                segment["assessment"] = None

            if progress_cb:

                progress_cb(
                    (i + 1) / total,
                    f"Analyzing sentence {i + 1}"
                )

        analysis = {

            "essay": essay_text,

            "engine": {
                "name": self.name,
                "kind": self.kind
            },

            "segments": segments,

            "summary": summarize_analysis(
                {
                    "segments": segments
                }
            )
        }

        return analysis


# ============================================================
# SUMMARY
# ============================================================

def summarize_analysis(analysis):

    segments = analysis.get(
        "segments",
        []
    )

    type_counts = {}

    effectiveness_counts = {
        "Effective": 0,
        "Adequate": 0,
        "Ineffective": 0
    }

    confidence_values = []

    score_values = []

    weakness_counts = {}

    for segment in segments:

        discourse_type = segment.get(
            "type",
            "Other"
        )

        type_counts[
            discourse_type
        ] = (
            type_counts.get(
                discourse_type,
                0
            ) + 1
        )

        effectiveness = segment.get(
            "effectiveness"
        )

        if effectiveness in effectiveness_counts:

            effectiveness_counts[
                effectiveness
            ] += 1

        confidence = segment.get(
            "confidence"
        )

        if confidence is not None:

            confidence_values.append(
                float(confidence)
            )

        assessment = (
            segment.get("assessment")
            or {}
        )

        score = assessment.get(
            "score"
        )

        if score is not None:

            score_values.append(
                float(score)
            )

        for weakness in assessment.get(
            "weakness_types",
            []
        ):

            weakness_counts[
                weakness
            ] = (
                weakness_counts.get(
                    weakness,
                    0
                ) + 1
            )

    claims = sum(
        type_counts.get(
            label,
            0
        )
        for label in [
            "Position",
            "Claim",
            "Counterclaim",
            "Rebuttal"
        ]
    )

    evidence = type_counts.get(
        "Evidence",
        0
    )

    reasoning = sum(
        1
        for segment in segments
        if (
            segment.get(
                "assessment"
            )
            and segment[
                "assessment"
            ].get(
                "has_reasoning",
                False
            )
        )
    )

    if score_values:

        avg_score = round(
            sum(score_values)
            / len(score_values),
            2
        )

    else:

        avg_score = None

    if confidence_values:

        average_confidence = round(
            sum(confidence_values)
            / len(confidence_values),
            4
        )

    else:

        average_confidence = None

    weak_arguments = sum(
        1
        for segment in segments
        if segment.get(
            "effectiveness"
        ) == "Ineffective"
    )

    return {

        "type_counts": type_counts,

        "effectiveness_counts":
            effectiveness_counts,

        "claims": claims,

        "evidence": evidence,

        "reasoning": reasoning,

        "assessed": len(segments),

        "avg_score": avg_score,

        "average_confidence":
            average_confidence,

        "weak_arguments":
            weak_arguments,

        "weakness_counts":
            weakness_counts
    }


# ============================================================
# SELECT WEAK ARGUMENTS
# ============================================================

def select_weak_arguments(
    analysis,
    threshold=6
):

    weak = []

    for segment in analysis.get(
        "segments",
        []
    ):

        assessment = (
            segment.get("assessment")
            or {}
        )

        score = assessment.get(
            "score"
        )

        effectiveness = segment.get(
            "effectiveness"
        )

        if (
            effectiveness == "Ineffective"
            or (
                score is not None
                and score < threshold
            )
        ):

            weak.append(segment)

    return weak


# ============================================================
# PREDICTOR SINGLETON
# ============================================================

_predictor = None


def load_predictor():

    global _predictor

    if _predictor is None:

        _predictor = MLPredictor()

    return _predictor


# ============================================================
# CREATE ANALYZER
# ============================================================

def create_analyzer():

    return ArgumentAnalyzer(
        load_predictor()
    )