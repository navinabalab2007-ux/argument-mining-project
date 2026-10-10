# AI-Powered Argument Mining and Socratic Feedback System

Analyzes student essays (argument types, quality scores) and turns the scores into **Socratic
questions** instead of giving the answer. Based on the idea of argument mining with small LLMs
(Llama 3.1 8B) from https://ellisalicante.org/publications/favero2025amllama/ . This is an independent
implementation, not the authors' code, and it does not reproduce the paper's numbers.

## Run it on Windows
1. Install Python 3.10, 3.11 or 3.12 from python.org (tick **Add python.exe to PATH**).
2. Extract the ZIP, open the folder, double-click **run.bat**.
3. First run installs packages (5-15 min). The browser opens the app. Next runs start in seconds.
4. In the app: *Essay Input* -> *Load sample essay* -> *Analyze essay*.

The default **Heuristic demo** engine is rule-based (not Llama) and works on any laptop. To use Llama:

1. Request access to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and accept the license.
2. Create a read token at huggingface.co/settings/tokens and paste it in the app sidebar
   (or set the `HF_TOKEN` environment variable).
3. Sidebar -> Engine mode -> *Llama pretrained (few-shot)* -> **Load engine**.
   Llama 3.1 8B needs about 16 GB of GPU/RAM, or about 6 GB VRAM with 4-bit loading (NVIDIA GPU).
   NVIDIA users can run `install_gpu_torch.bat` for the CUDA build of PyTorch.
   If your laptop is too small, type a smaller model ID such as `meta-llama/Llama-3.2-3B-Instruct`.

## Fine-tuning (LoRA / QLoRA)
* `train_llama_sample.bat`: tiny test run on the built-in sample (proves the pipeline works; the adapter is
  not a useful model).
* Real training: put Kaggle's *Feedback Prize - Predicting Effective Arguments* `train.csv` in
  `data\feedback-prize\` and run
  `.venv\Scripts\python train.py --dataset feedback-prize --max-train-samples 2000`.
* The adapter is saved to `models\llama_lora_adapter` with `training_meta.json`; the app offers the
  *fine-tuned* engine only when that trained adapter exists. Checkpoints go to `checkpoints\`.
* `train_bert_sample.bat` trains the BERT baseline on the same training split (`--target bert`).

## Evaluation and comparison
Model Evaluation page (or `evaluate.py`, `evaluate_all.bat`): accuracy, macro/weighted precision, recall, F1,
confusion matrices, predictions-vs-ground-truth table, saved under `outputs\evaluation\<run>\`.
The comparison table lists only runs that were actually measured and warns when test sets differ.

## Files
`app.py` UI · `config.py` settings · `data_loader.py` CSV loading/validation/splits · `preprocessing.py`
segmentation + prompts · `model.py` engines (heuristic / Llama / LoRA) · `heuristic_engine.py` rule-based demo ·
`predict.py` classification + assessment · `socratic_feedback.py` Score-to-Socratic sessions · `feedback.py`
suggestions + before/after comparison · `train.py` LoRA/BERT training · `baseline.py` BERT ·
`evaluate.py` metrics · `utils.py` · `tests/smoke_test.py` · `sample_essays/` · `data/sample/sample_train.csv`
(36 hand-written spans, NOT from Feedback Prize) · `.bat` helpers.

## Score-to-Socratic feedback
Weak argument + assessment -> question generated from the weakness -> student answers in the chat -> follow-up built
from the answer and full conversation -> repeat -> student submits a revision -> reassessment and score change ->
suggestions unlock (hints, never a rewrite). Questions come from Llama when a Llama engine is active; with the demo
engine they are templates filled with the student's own words and are labelled "template".

## Known limits
* The dataset labels are Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement and
  effectiveness (Ineffective/Adequate/Effective). It has **no "reasoning" label**; reasoning is reported from the assessment.
* Scores 1-10 and explanations are model output, not verified labels. The 1-10 scale has no ground truth.
* Essays are split into sentences (not gold spans), so real essay results differ from span-level evaluation.
* Llama on CPU is slow (minutes per essay).

## What was tested (in the build environment, Linux, Python 3.12, no internet)
Run: `python tests/smoke_test.py` (34 checks passed): dataset loading/validation/splits, segmentation, heuristic
classification and assessment, highlighting HTML, full Socratic workflow (questions, follow-ups, revision, suggestion
lock/unlock, JSON export), evaluation outputs, and the LLM code paths using a scripted stub (parsing, fallbacks).
`train.py` and `evaluate.py --engine bert` were run and fail gracefully with a clear message when torch is missing.
**Not tested:** the Streamlit UI (streamlit not installable there; app.py only compiled), real Llama 3.1 8B download and
inference, LoRA/QLoRA training, BERT training, bitsandbytes, the .bat files on Windows. Expect to report any first-run
issues from those parts.
