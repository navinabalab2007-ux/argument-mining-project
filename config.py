"""Central configuration for the Argument Mining + Socratic Feedback project."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Folders (created automatically by utils.ensure_dirs) ----
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
FEEDBACK_PRIZE_DIR = Path(os.environ.get("FEEDBACK_PRIZE_DIR", DATA_DIR / "feedback-prize"))
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
SAMPLE_ESSAYS_DIR = BASE_DIR / "sample_essays"
ALL_DIRS = [DATA_DIR, SAMPLE_DIR, FEEDBACK_PRIZE_DIR, MODELS_DIR, CHECKPOINTS_DIR,
            OUTPUTS_DIR, EVAL_DIR, SESSIONS_DIR, SAMPLE_ESSAYS_DIR]

SAMPLE_CSV = SAMPLE_DIR / "sample_train.csv"
LLAMA_ADAPTER_DIR = MODELS_DIR / "llama_lora_adapter"
BERT_DIR = MODELS_DIR / "bert_baseline"

# ---- Models ----
# Llama 3.1 8B is gated on Hugging Face: request access on the model page and log in
# (huggingface-cli login) or set the HF_TOKEN environment variable.
DEFAULT_MODEL_ID = os.environ.get("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
BERT_MODEL_NAME = os.environ.get("BERT_MODEL_NAME", "bert-base-uncased")

# ---- Labels (Feedback Prize - Predicting Effective Arguments) ----
DISCOURSE_TYPES = ["Lead", "Position", "Claim", "Counterclaim", "Rebuttal",
                   "Evidence", "Concluding Statement"]
SEGMENT_TYPES = DISCOURSE_TYPES + ["Other"]          # "Other" = non-argumentative text
EFFECTIVENESS_LABELS = ["Ineffective", "Adequate", "Effective"]
NOT_APPLICABLE = "N/A"

CLAIM_GROUP = ["Position", "Claim", "Counterclaim", "Rebuttal"]
EVIDENCE_GROUP = ["Evidence"]
STRUCTURE_GROUP = ["Lead", "Concluding Statement"]
ASSESSED_TYPES = CLAIM_GROUP + EVIDENCE_GROUP        # types that receive a quality assessment

TYPE_COLORS = {
    "Lead": "#E1D5F5", "Position": "#BFDBFE", "Claim": "#93C5FD",
    "Counterclaim": "#FDE68A", "Rebuttal": "#FDBA74", "Evidence": "#BBF7D0",
    "Concluding Statement": "#F5D0FE", "Other": "transparent",
}
REASONING_CUE_COLOR = "#FCA5A5"
REASONING_CUES = (r"\b(because|therefore|thus|as a result|consequently|which means|this means|"
                  r"so that|since|leads? to|that is why|due to|for this reason|in order to)\b")

# ---- Essay limits ----
MIN_ESSAY_WORDS = 20
MAX_ESSAY_WORDS = 3000
MIN_SEGMENT_WORDS = 4
MAX_SEGMENT_WORDS_FOR_MODEL = 250

# ---- Assessment / Socratic behaviour ----
WEAK_SCORE_THRESHOLD = 6          # arguments scoring <= this are treated as weak
MAX_LLM_ASSESSMENTS = 12          # cap on per-essay LLM assessments (speed on laptops)
SOCRATIC_ROUNDS_BEFORE_REVISION_HINT = 3
MAX_STUDENT_ANSWER_CHARS = 2000
MIN_STUDENT_ANSWER_CHARS = 3

# ---- Generation defaults ----
CLASSIFY_MAX_NEW_TOKENS = 16
ASSESS_MAX_NEW_TOKENS = 320
QUESTION_MAX_NEW_TOKENS = 90
SUGGEST_MAX_NEW_TOKENS = 300

# ---- Training defaults ----
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
