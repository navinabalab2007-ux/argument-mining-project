"""Streamlit app: Argument Mining + Score-to-Socratic Feedback.

Start with run.bat (Windows) or:
python -m streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import config
from data_loader import DataLoaderError, get_dataset
from evaluate import comparison_table, evaluate_predictor, get_eval_frame, load_runs
from model import ModelError, adapter_info, create_engine
from predict import WEAKNESS_LABELS, ArgumentAnalyzer, select_weak_arguments
from socratic_feedback import SocraticSession
from utils import (
    decode_upload,
    ensure_dirs,
    hardware_summary,
    legend_html,
    render_highlighted_html,
)

ensure_dirs()

st.set_page_config(
    page_title="Argument Mining & Socratic Feedback",
    page_icon="🧠",
    layout="wide",
)

PAGES = [
    "Home",
    "Essay Input",
    "Argument Classification",
    "Quality Assessment",
    "Socratic Feedback",
    "Improvement Suggestions",
    "Dashboard",
    "Model Evaluation",
]

SOURCE_NOTE = {
    "llm": "Llama",
    "llm-fewshot": "Llama few-shot",
    "llm-finetuned": "Llama fine-tuned",
    "heuristic": "rule-based demo",
    "heuristic-fallback": "rule-based (model output unusable)",
    "template": "template",
    "template-fallback": "template (model output unusable)",
    "rule": "rule",
}


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

def init_state():
    if "analyzer" not in st.session_state:
        st.session_state.engine = create_engine("heuristic")
        st.session_state.analyzer = ArgumentAnalyzer(
            st.session_state.engine
        )

    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("sessions", {})
    st.session_state.setdefault("essay_text", "")
    st.session_state.setdefault("essay_box", "")
    st.session_state.setdefault("uploaded_name", None)
    st.session_state.setdefault(
        "dataset_dir", str(config.FEEDBACK_PRIZE_DIR)
    )
    st.session_state.setdefault("eval_result", None)


def sample_essay_text():
    path = config.SAMPLE_ESSAYS_DIR / "sample_essay.txt"

    if path.exists():
        return path.read_text(encoding="utf-8")

    return ""


def need_analysis():
    if st.session_state.analysis is None:
        st.info(
            "Please analyze an essay first "
            "(page: **Essay Input**)."
        )
        return False

    return True


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

def sidebar():
    st.sidebar.title("🧠 Argument Coach")

    page = st.sidebar.radio(
        "Go to",
        PAGES,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Engine")

    engine = st.session_state.engine

    st.sidebar.caption(f"Active: **{engine.name}**")

    mode_label = st.sidebar.selectbox(
        "Engine mode",
        [
            "Heuristic demo (no download)",
            "Llama pretrained (few-shot)",
            "Llama fine-tuned (LoRA adapter)",
        ],
    )

    mode = {
        "Heuristic demo (no download)": "heuristic",
        "Llama pretrained (few-shot)": "pretrained",
        "Llama fine-tuned (LoRA adapter)": "finetuned",
    }[mode_label]

    model_id = config.DEFAULT_MODEL_ID
    adapter_dir = None
    quant = "auto"
    token = None
    force = False

    if mode != "heuristic":
        model_id = st.sidebar.text_input(
            "Hugging Face model ID or local folder",
            config.DEFAULT_MODEL_ID,
        )

        quant = st.sidebar.selectbox(
            "Quantization",
            ["auto", "4bit", "none"],
            help=(
                "4bit needs an NVIDIA GPU + bitsandbytes "
                "and saves memory."
            ),
        )

        token = st.sidebar.text_input(
            "HF access token (optional)",
            type="password",
            help=(
                "Needed for gated Llama models unless you "
                "ran huggingface-cli login. Not stored."
            ),
        )

        force = st.sidebar.checkbox(
            "Try anyway if memory looks too small"
        )

    if mode == "finetuned":
        adapter_dir = st.sidebar.text_input(
            "LoRA adapter folder",
            str(config.LLAMA_ADAPTER_DIR),
        )

        meta = adapter_info(adapter_dir)

        if meta:
            st.sidebar.success(
                f"Trained adapter found "
                f"({meta.get('n_train_examples')} examples, "
                f"source: {meta.get('dataset_source')})."
            )
        else:
            st.sidebar.warning(
                "No trained adapter found yet. "
                "Run train_llama_sample.bat first."
            )

    if st.sidebar.button("Load engine", use_container_width=True):
        with st.spinner("Loading model..."):
            try:
                new_engine = create_engine(
                    mode,
                    model_id,
                    adapter_dir,
                    quant,
                    token or None,
                    force,
                )

                st.session_state.engine = new_engine
                st.session_state.analyzer = ArgumentAnalyzer(
                    new_engine
                )
                st.session_state.analysis = None
                st.session_state.sessions = {}

                for note in getattr(new_engine, "notes", []):
                    st.sidebar.info(note)

                st.sidebar.success(
                    "Engine ready. Re-run the analysis to use it."
                )

            except ModelError as exc:
                st.sidebar.error(str(exc))

            except Exception as exc:
                st.sidebar.error(
                    f"Unexpected error while loading: {exc}"
                )

    with st.sidebar.expander("Hardware"):
        hw = hardware_summary()
        st.write({key: value for key, value in hw.items()})

    return page


# ------------------------------------------------------------
# Home page
# ------------------------------------------------------------

def page_home():
    st.title("AI-Powered Argument Mining & Socratic Feedback")

    st.write(
        "Analyze a student essay, see which sentences are claims, "
        "evidence, and other argument parts, check how strong each "
        "argument is, and improve your writing through guided "
        "**Socratic questions** instead of being handed the answer."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Active engine",
        st.session_state.engine.kind,
    )

    col2.metric(
        "Essay analyzed",
        "yes" if st.session_state.analysis else "no",
    )

    col3.metric(
        "Socratic sessions",
        len(st.session_state.sessions),
    )

    st.subheader("How to use it")

    st.markdown(
        "1. **Essay Input**: paste or upload an essay, click *Analyze*.\n"
        "2. Look at **Argument Classification** and **Quality Assessment**.\n"
        "3. Open **Socratic Feedback**, pick a weak argument, "
        "and answer the questions.\n"
        "4. Submit a revised argument to see the new score. "
        "Then open **Improvement Suggestions**.\n"
        "5. **Dashboard** and **Model Evaluation** show statistics "
        "and measured model results."
    )

    st.subheader("Honest notes")

    st.warning(
        "The default **Heuristic demo** engine is rule-based. "
        "It is *not* Llama and is not a trained model; it exists "
        "so the app runs on any laptop. For real Llama 3.1 8B "
        "results choose an engine in the sidebar."
    )

    st.markdown(
        "- Scores are **model predictions**, not verified ground truth. "
        "Ground-truth labels only exist in the Feedback Prize dataset "
        "and are used on the *Model Evaluation* page.\n"
        "- The dataset has no separate 'reasoning' label. "
        "The dashboard's reasoning count = arguments the assessment "
        "says contain an explained reason.\n"
        "- The engine is called *fine-tuned* only when a trained "
        "adapter really exists on disk."
    )


# ------------------------------------------------------------
# Essay input page
# ------------------------------------------------------------

def page_input():
    st.header("Essay Input")

    col1, col2 = st.columns([3, 1])

    # Upload and sample controls
    with col2:
        if st.button("Load sample essay", use_container_width=True):
            sample = sample_essay_text()

            st.session_state.essay_text = sample
            st.session_state.essay_box = sample
            st.session_state.uploaded_name = None

            st.rerun()

        uploaded_file = st.file_uploader(
            "Upload a .txt file",
            type=["txt", "md"],
            key="essay_uploader",
        )

        if uploaded_file is not None:
            try:
                uploaded_text = decode_upload(
                    uploaded_file.getvalue()
                )

                if (
                    st.session_state.get("uploaded_name")
                    != uploaded_file.name
                    or uploaded_text
                    != st.session_state.essay_text
                ):
                    st.session_state.essay_text = uploaded_text
                    st.session_state.essay_box = uploaded_text
                    st.session_state.uploaded_name = (
                        uploaded_file.name
                    )

                st.success(f"Loaded {uploaded_file.name}")

            except ValueError as exc:
                st.error(str(exc))

    # Essay text area
    with col1:
        text = st.text_area(
            "Paste the essay here (one paragraph per line break)",
            height=340,
            key="essay_box",
        )

        st.session_state.essay_text = text

    if st.session_state.engine.is_llm:
        st.info(
            "Llama classifies every sentence and then assesses "
            "arguments. On a laptop this can take several minutes."
        )

    # Analyze button
    if st.button("Analyze essay", type="primary"):
        essay = st.session_state.get("essay_box", "").strip()

        if not essay:
            st.error(
                "The essay is empty. Paste an essay or upload a .txt file."
            )
            return

        bar = st.progress(0.0, text="Starting...")

        try:
            result = st.session_state.analyzer.analyze_essay(
                essay,
                progress_cb=lambda fraction, message: bar.progress(
                    min(1.0, fraction),
                    text=message,
                ),
            )

            st.session_state.analysis = result
            st.session_state.sessions = {}

            bar.progress(1.0, text="Done")
            st.success(
                "Analysis complete. Open the other pages from the sidebar."
            )

        except ValueError as exc:
            bar.empty()
            st.error(str(exc))

        except ModelError as exc:
            bar.empty()
            st.error(str(exc))

        except Exception as exc:
            bar.empty()
            st.error(f"Something went wrong: {exc}")


# ------------------------------------------------------------
# Argument classification
# ------------------------------------------------------------

def page_classification():
    st.header("Argument Classification")

    if not need_analysis():
        return

    analysis = st.session_state.analysis

    st.caption(
        f"Engine: {analysis['engine']['name']}. "
        "Labels below are model **predictions**, not ground truth."
    )

    show_reasoning = st.checkbox(
        "Show reasoning cues "
        "(rule-based underline of words like 'because')",
        True,
    )

    st.markdown(
        legend_html(),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_highlighted_html(
            analysis["essay"],
            analysis["segments"],
            show_reasoning,
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Span table")

    rows = [
        {
            "#": segment["idx"] + 1,
            "Predicted type": segment["type"],
            "Predicted effectiveness": segment["effectiveness"],
            "Source": SOURCE_NOTE.get(
                segment["source"],
                segment["source"],
            ),
            "Text": segment["text"],
        }
        for segment in analysis["segments"]
    ]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    unparsed = [
        segment
        for segment in analysis["segments"]
        if segment["source"].endswith("unparsed")
    ]

    if unparsed:
        st.warning(
            f"{len(unparsed)} sentence(s) got an unreadable model "
            "answer and were set to 'Other'."
        )


# ------------------------------------------------------------
# Quality assessment
# ------------------------------------------------------------

def page_quality():
    st.header("Argument Quality Assessment")

    if not need_analysis():
        return

    analysis = st.session_state.analysis

    st.caption(
        "Scores (1-10) and explanations are **model estimates**, "
        "not verified ground truth."
    )

    assessed = [
        segment
        for segment in analysis["segments"]
        if segment.get("assessment")
    ]

    if not assessed:
        st.info("No arguments were assessed.")

    for segment in assessed:
        assessment = segment["assessment"]

        icon = (
            "🟢" if assessment["score"] >= 7
            else "🟡" if assessment["score"] >= 4
            else "🔴"
        )

        with st.expander(
            f"{icon} {assessment['score']}/10 · "
            f"{segment['type']} · {segment['text'][:80]}"
        ):
            st.write(f"**Argument:** {segment['text']}")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Score",
                f"{assessment['score']}/10",
            )

            col2.metric(
                "Effectiveness",
                assessment["effectiveness"],
            )

            col3.metric(
                "Assessed by",
                SOURCE_NOTE.get(
                    assessment["source"],
                    assessment["source"],
                ),
            )

            st.write(
                "**Why this score:** "
                + (assessment["explanation"] or "n/a")
            )

            col_strength, col_weakness = st.columns(2)

            col_strength.markdown(
                "**Strengths**\n"
                + (
                    "\n".join(
                        f"- {item}"
                        for item in assessment["strengths"]
                    )
                    or "- none identified"
                )
            )

            col_weakness.markdown(
                "**Weaknesses**\n"
                + (
                    "\n".join(
                        f"- {item}"
                        for item in assessment["weaknesses"]
                    )
                    or "- none identified"
                )
            )

            if assessment["weakness_types"]:
                st.write(
                    "Weakness tags: "
                    + ", ".join(
                        WEAKNESS_LABELS.get(item, item)
                        for item in assessment["weakness_types"]
                    )
                )

    skipped = [
        segment
        for segment in analysis["segments"]
        if segment.get("assessment_skipped")
    ]

    if skipped:
        st.info(
            f"{len(skipped)} argument(s) were not assessed to keep "
            f"the run fast (limit: {config.MAX_LLM_ASSESSMENTS})."
        )


# ------------------------------------------------------------
# Socratic feedback
# ------------------------------------------------------------

def _seg_label(segment):
    return (
        f"{segment['assessment']['score']}/10 · "
        f"{segment['type']} · {segment['text'][:70]}"
    )


def page_socratic():
    st.header("Socratic Feedback")

    if not need_analysis():
        return

    analysis = st.session_state.analysis
    analyzer = st.session_state.analyzer

    candidates = (
        select_weak_arguments(analysis)
        or [
            segment
            for segment in analysis["segments"]
            if segment.get("assessment")
        ]
    )

    if not candidates:
        st.info("No assessed arguments are available.")
        return

    if not select_weak_arguments(analysis):
        st.success(
            "No weak arguments were found. "
            "You can still practice with any argument."
        )

    st.write(
        "Pick an argument to work on. The tutor will ask questions "
        "and will **not** give you the answer."
    )

    idx_map = {
        segment["idx"]: segment
        for segment in candidates
    }

    selected = st.selectbox(
        "Argument",
        list(idx_map),
        format_func=lambda index: _seg_label(idx_map[index]),
    )

    segment = idx_map[selected]
    sessions = st.session_state.sessions

    if selected not in sessions:
        st.write(f"**Original argument:** {segment['text']}")

        st.write(
            f"**Assessment:** "
            f"{segment['assessment']['score']}/10, "
            f"{segment['assessment']['effectiveness']}"
        )

        if st.button(
            "Start Socratic session",
            type="primary",
        ):
            try:
                with st.spinner("Preparing a question..."):
                    session = SocraticSession(
                        analyzer,
                        analysis["essay"],
                        segment,
                    )
                    session.start()

                sessions[selected] = session
                st.rerun()

            except (ModelError, ValueError) as exc:
                st.error(str(exc))

        return

    session = sessions[selected]

    answer = st.chat_input(
        "Type your answer to the tutor's question..."
    )

    if answer:
        try:
            with st.spinner("Thinking of a follow-up question..."):
                session.reply(answer)

        except (ModelError, ValueError) as exc:
            st.error(str(exc))

    left, right = st.columns([3, 2])

    with right:
        st.markdown("#### Argument status")

        st.write(
            f"**Original ({session.original_assessment['score']}/10):** "
            f"{session.original_text}"
        )

        if session.revisions:
            st.write(
                f"**Latest version "
                f"({session.current_assessment['score']}/10):** "
                f"{session.current_text}"
            )

        st.write(
            "**Weakness in focus:** "
            + WEAKNESS_LABELS.get(
                session.weakness,
                "Could be developed further",
            )
        )

        for weakness in session.original_assessment.get(
            "weaknesses", []
        ):
            st.caption("• " + weakness)

        if session.suggest_revision:
            st.info(
                "You have thought about this a lot. "
                "Try rewriting the argument below when you feel ready."
            )

        st.markdown("#### Revise your argument")

        new_text = st.text_area(
            "Your improved version",
            value=session.current_text,
            height=120,
            key=f"rev_{selected}_{len(session.revisions)}",
        )

        if st.button("Submit revision & reassess"):
            try:
                with st.spinner("Reassessing..."):
                    session.revise(new_text)

                st.rerun()

            except (ModelError, ValueError) as exc:
                st.error(str(exc))

        if session.revisions:
            comparison = session.revisions[-1]["comparison"]

            st.metric(
                "Score after revision",
                f"{comparison['new_score']}/10",
                delta=comparison["delta"],
            )

            st.write(comparison["message"])

            if comparison["resolved_labels"]:
                st.success(
                    "Resolved: "
                    + ", ".join(comparison["resolved_labels"])
                )

            if comparison["remaining_labels"]:
                st.warning(
                    "Still to work on: "
                    + ", ".join(comparison["remaining_labels"])
                )

            with st.expander("Reassessment details"):
                assessment = session.current_assessment
                st.write(assessment["explanation"])
                st.caption(
                    "Assessed by: "
                    + SOURCE_NOTE.get(
                        assessment["source"],
                        assessment["source"],
                    )
                )

    with left:
        st.markdown("#### Conversation")

        for message in session.history:
            role = (
                "assistant"
                if message["role"] == "ai"
                else "user"
            )

            with st.chat_message(role):
                st.write(message["content"])

                if message["role"] == "ai":
                    st.caption(
                        "question source: "
                        + SOURCE_NOTE.get(
                            message["source"],
                            message["source"],
                        )
                    )

        if session.can_view_suggestions:
            st.caption(
                "Suggestions are now unlocked on the "
                "*Improvement Suggestions* page."
            )

    with st.expander(
        "Feedback card "
        "(original → weakness → questions → responses → suggestion)"
    ):
        card = session.feedback_card()

        st.write(
            f"**Original argument "
            f"({card['argument_type']}, "
            f"{card['original_score']}/10):** "
            f"{card['original_argument']}"
        )

        st.write(
            f"**Identified weakness:** "
            f"{card['identified_weakness']}"
        )

        for index, exchange in enumerate(
            card["exchanges"], 1
        ):
            st.write(
                f"**Q{index}:** {exchange['question']}"
            )
            st.write(
                "**Your response:** "
                + (
                    exchange["student_response"]
                    or "(not answered yet)"
                )
            )

        if card["revised_argument"]:
            st.write(
                f"**Revised argument "
                f"({card['latest_score']}/10):** "
                f"{card['revised_argument']}"
            )

        suggestion = card["suggested_improvement"]

        st.write(
            "**Suggested improvement:** "
            + (
                suggestion
                if isinstance(suggestion, str)
                else " / ".join(suggestion)
            )
        )

    if st.button("Save this session to outputs/sessions"):
        st.success(f"Saved: {session.save()}")


# ------------------------------------------------------------
# Improvement suggestions
# ------------------------------------------------------------

def page_suggestions():
    st.header("Improvement Suggestions")

    if not need_analysis():
        return

    sessions = st.session_state.sessions

    if not sessions:
        st.info(
            "Start a Socratic session first. Suggestions unlock "
            "after you try to improve an argument."
        )
        return

    st.caption(
        "Suggestions are hints, not rewrites. "
        "They unlock after you answer a question or revise."
    )

    for index, session in sessions.items():
        with st.expander(
            f"{session.arg_type}: {session.current_text[:80]}",
            expanded=session.can_view_suggestions,
        ):
            if not session.can_view_suggestions:
                st.warning(
                    "Locked. Answer at least one Socratic question "
                    "or submit a revision first."
                )
                continue

            if st.button(
                "Show suggestions",
                key=f"sug_{index}",
            ):
                try:
                    with st.spinner("Preparing suggestions..."):
                        session.get_suggestions()

                except (ModelError, PermissionError) as exc:
                    st.error(str(exc))

            if session.suggestion_result:
                for tip in session.suggestion_result["suggestions"]:
                    st.write("• " + tip)

                st.caption(
                    "source: "
                    + SOURCE_NOTE.get(
                        session.suggestion_result["source"],
                        session.suggestion_result["source"],
                    )
                )


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------

def page_dashboard():
    st.header("Argument Strength Dashboard")

    if not need_analysis():
        return

    analysis = st.session_state.analysis
    summary = analysis["summary"]

    columns = st.columns(5)

    columns[0].metric(
        "Claims (position/claim/counter/rebuttal)",
        summary["claims"],
    )

    columns[1].metric(
        "Evidence spans",
        summary["evidence"],
    )

    columns[2].metric(
        "Reasoning components",
        summary["reasoning"],
    )

    columns[3].metric(
        "Average score",
        (
            f"{summary['avg_score']}/10"
            if summary["avg_score"] is not None
            else "n/a"
        ),
    )

    columns[4].metric(
        "Weak arguments",
        summary["weak_arguments"],
    )

    st.caption(
        "Reasoning components = assessed arguments where the "
        "assessment found an explained reason."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Predicted argument types")
        st.bar_chart(
            pd.Series(
                summary["type_counts"],
                name="count",
            )
        )

    with right:
        st.subheader("Detected weaknesses")

        if summary["weakness_counts"]:
            st.bar_chart(
                pd.Series(
                    {
                        WEAKNESS_LABELS.get(key, key): value
                        for key, value
                        in summary["weakness_counts"].items()
                    },
                    name="count",
                )
            )
        else:
            st.write("No weaknesses detected.")

    assessed = [
        segment
        for segment in analysis["segments"]
        if segment.get("assessment")
    ]

    if assessed:
        st.subheader("Score per argument")

        df = pd.DataFrame(
            {
                "argument": [
                    f"{segment['idx'] + 1}. {segment['type']}"
                    for segment in assessed
                ],
                "score": [
                    segment["assessment"]["score"]
                    for segment in assessed
                ],
            }
        ).set_index("argument")

        st.bar_chart(df)

        effectiveness = pd.Series(
            [
                segment["assessment"]["effectiveness"]
                for segment in assessed
            ]
        ).value_counts()

        st.subheader("Effectiveness distribution")
        st.bar_chart(effectiveness)

    revised = [
        session
        for session in st.session_state.sessions.values()
        if session.revisions
    ]

    if revised:
        st.subheader("Socratic progress")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "argument": session.original_text[:70],
                        "before": session.original_assessment["score"],
                        "after": session.current_assessment["score"],
                        "rounds": session.rounds,
                    }
                    for session in revised
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

def page_evaluation():
    st.header("Model Evaluation")

    st.write(
        "Compares predictions with **ground-truth labels** "
        "from the dataset. Only measured results are shown."
    )

    source_label = st.radio(
        "Dataset",
        [
            "Built-in sample (tiny, hand-written demo)",
            "Feedback Prize CSV folder",
        ],
    )

    source = (
        "sample"
        if source_label.startswith("Built-in")
        else "feedback-prize"
    )

    directory = None

    if source == "feedback-prize":
        st.session_state.dataset_dir = st.text_input(
            "Dataset folder (contains train.csv)",
            st.session_state.dataset_dir,
        )

        directory = st.session_state.dataset_dir

        try:
            df, report = get_dataset(
                "feedback-prize",
                directory,
            )

            st.success(
                f"Dataset OK: {report['rows_out']} valid rows "
                f"(dropped {report['dropped_invalid_labels']} invalid, "
                f"{report['dropped_duplicates']} duplicates)."
            )

        except DataLoaderError as exc:
            st.error(str(exc))
            directory = None

    else:
        st.info(
            "The sample contains 36 hand-written spans "
            "(not from Feedback Prize). Scores only show "
            "the pipeline works."
        )

    col1, col2 = st.columns(2)

    split = col1.selectbox(
        "Split",
        ["test", "all"],
        index=1 if source == "sample" else 0,
        help=(
            "'test' = essay-grouped hold-out split "
            "(same split used by train.py)."
        ),
    )

    default_n = (
        30 if st.session_state.engine.is_llm else 500
    )

    max_n = col2.number_input(
        "Max samples",
        1,
        100000,
        default_n,
    )

    can_run = source == "sample" or directory is not None
    engine = st.session_state.engine

    if st.button(
        f"Evaluate current engine: {engine.kind}",
        disabled=not can_run,
        type="primary",
    ):
        bar = st.progress(0.0, text="Evaluating...")

        try:
            frame, _ = get_eval_frame(
                source,
                directory,
                split,
                int(max_n),
            )

            label = (
                engine.kind
                if engine.kind != "pretrained"
                else "llama_pretrained"
            )

            if engine.kind == "finetuned":
                label = "llama_finetuned"

            result = evaluate_predictor(
                st.session_state.analyzer,
                frame,
                label,
                source,
                split,
                progress_cb=lambda fraction, message: bar.progress(
                    min(1.0, fraction),
                    text=message,
                ),
            )

            st.session_state.eval_result = result
            bar.progress(1.0, text="Done")

        except (DataLoaderError, ModelError, ValueError) as exc:
            bar.empty()
            st.error(str(exc))

    if (Path(config.BERT_DIR) / "meta.json").exists():
        if st.button(
            "Evaluate BERT baseline on the same data",
            disabled=not can_run,
        ):
            try:
                from baseline import BertBaseline

                frame, _ = get_eval_frame(
                    source,
                    directory,
                    split,
                    int(max_n),
                )

                with st.spinner("Running BERT..."):
                    st.session_state.eval_result = evaluate_predictor(
                        BertBaseline.load(),
                        frame,
                        "bert",
                        source,
                        split,
                    )

            except Exception as exc:
                st.error(str(exc))

    else:
        st.caption(
            "BERT baseline not trained yet: run "
            "train_bert_sample.bat (or python train.py --target bert)."
        )

    result = st.session_state.eval_result

    if result:
        st.subheader(f"Latest result: {result['run_id']}")

        st.caption(
            f"Model: {result['model']} · "
            f"n={result['n_samples']} · {result['note']}"
        )

        for key, title in (
            ("type", "Argument type"),
            ("effectiveness", "Effectiveness"),
        ):
            metrics = result[key]

            st.markdown(f"**{title}**")

            columns = st.columns(5)

            columns[0].metric(
                "Accuracy",
                f"{metrics['accuracy']:.3f}",
            )

            columns[1].metric(
                "Precision (macro)",
                f"{metrics['precision_macro']:.3f}",
            )

            columns[2].metric(
                "Recall (macro)",
                f"{metrics['recall_macro']:.3f}",
            )

            columns[3].metric(
                "F1 (macro)",
                f"{metrics['f1_macro']:.3f}",
            )

            columns[4].metric(
                "F1 (weighted)",
                f"{metrics['f1_weighted']:.3f}",
            )

            image_path = Path(result["output_dir"]) / (
                "confusion_type.png"
                if key == "type"
                else "confusion_effectiveness.png"
            )

            if image_path.exists():
                st.image(
                    str(image_path),
                    caption=f"Confusion matrix: {title}",
                )

        predictions = pd.read_csv(
            Path(result["output_dir"]) / "predictions.csv"
        )

        st.markdown("**Predictions vs ground truth**")

        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Saved in: {result['output_dir']}")

    st.subheader("Model comparison (all saved runs)")

    table = comparison_table(load_runs())

    if table.empty:
        st.info("No evaluation runs saved yet.")
    else:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        if table["test_set_id"].nunique() > 1:
            st.warning(
                "Runs with different 'test_set_id' were measured "
                "on different data and are not directly comparable."
            )
        else:
            st.success(
                "All listed runs used the same test data."
            )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    init_state()

    page = sidebar()

    pages = {
        "Home": page_home,
        "Essay Input": page_input,
        "Argument Classification": page_classification,
        "Quality Assessment": page_quality,
        "Socratic Feedback": page_socratic,
        "Improvement Suggestions": page_suggestions,
        "Dashboard": page_dashboard,
        "Model Evaluation": page_evaluation,
    }

    pages[page]()


main()