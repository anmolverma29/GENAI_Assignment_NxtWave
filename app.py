"""Streamlit Interactive Application for the Self-Evaluating Lesson Content Generator."""

import json
import streamlit as st
from pathlib import Path
from src.config import Config
from src.models import CheckpointID, PipelineOutput, RubricEvaluation
from src.pipeline import run_pipeline
from src.memory import MemoryStore
from src.llm_provider import GeminiProvider, OpenAIProvider, MockProvider

st.set_page_config(
    page_title="Self-Evaluating Lesson Content Generator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .checkpoint-pass {
        background-color: #e8f5e9;
        border-radius: 6px;
        padding: 10px 14px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 8px;
    }
    .checkpoint-fail {
        background-color: #ffebee;
        border-radius: 6px;
        padding: 10px 14px;
        border-left: 5px solid #c62828;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Control Panel")

topic_option = st.sidebar.selectbox(
    "Lesson Topic",
    [
        "Introduction to RAG (Retrieval-Augmented Generation)",
        "Vector Embeddings for Beginners",
        "Fine-Tuning vs RAG: Which One When?",
        "Custom Topic..."
    ]
)

if topic_option == "Custom Topic...":
    topic = st.sidebar.text_input("Enter Topic", "Introduction to Prompt Engineering")
else:
    topic = topic_option

provider_choice = st.sidebar.selectbox(
    "LLM Execution Provider",
    ["Mock Simulation (Guaranteed Iterative Demo)", "Google Gemini API", "OpenAI API"]
)

api_key = ""
provider_key = "mock"
if provider_choice == "Google Gemini API":
    provider_key = "gemini"
    api_key = st.sidebar.text_input("Gemini API Key", value=Config.GEMINI_API_KEY, type="password")
elif provider_choice == "OpenAI API":
    provider_key = "openai"
    api_key = st.sidebar.text_input("OpenAI API Key", value=Config.OPENAI_API_KEY, type="password")

max_retries = st.sidebar.slider("Max Regeneration Retries", min_value=1, max_value=3, value=2)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 Pipeline Core Rules
1. **Target**: Absolute zero-background beginners.
2. **Rubric**: 6 hard pass/fail checkpoints.
3. **No Partial Credit**: 1 fail triggers full retry.
4. **Rejection Audit**: Logs exact reasons & remediations.
5. **Self-Evolving Memory**: Persists rules across runs.
""")

memory = MemoryStore()

# Main Header
st.title("🏗️ Self-Evaluating Lesson Content Generator")
st.caption("Agentic generate ➔ evaluate ➔ regenerate loop powered by LangGraph, Hard Binary Rubrics, and Cross-Run Memory.")

# Architecture Expander
with st.expander("🗺️ View LangGraph Agentic Workflow Architecture"):
    st.markdown("""
    ```mermaid
    graph TD
        A[Start: Topic Input] --> B[Initialize Memory & Evolved Rules]
        B --> C[Generate Lesson Draft]
        C --> D[Evaluate: 6 Hard Pass/Fail Checkpoints]
        D -->|All 6 Pass| E[Evolve Guidelines]
        D -->|Any Fail & Retries Left| F[Record Rejection & Formulate Remediation]
        F -->|Inject Feedback| C
        D -->|Any Fail & Max Retries Hit| G[Force Terminate & Audit Log]
        E --> H[Finalize & Save Outputs]
        G --> H
        H --> I[End: Ship Ready Content]
    ```
    """)

# Action Button
if st.button("🚀 Run Agentic Generation Pipeline", type="primary", use_container_width=True):
    with st.spinner("Executing agentic loop with rubric evaluations..."):
        custom_llm = None
        if provider_key == "gemini":
            if not api_key:
                st.error("Please provide a Gemini API Key or switch to Mock Simulation.")
                st.stop()
            custom_llm = GeminiProvider(api_key=api_key)
        elif provider_key == "openai":
            if not api_key:
                st.error("Please provide an OpenAI API Key or switch to Mock Simulation.")
                st.stop()
            custom_llm = OpenAIProvider(api_key=api_key)
        else:
            custom_llm = MockProvider()

        output = run_pipeline(
            topic=topic,
            max_retries=max_retries,
            custom_llm=custom_llm,
            memory_store=memory
        )
        st.session_state["output"] = output
        st.session_state["active_topic"] = topic

if "output" in st.session_state:
    output: PipelineOutput = st.session_state["output"]

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Status", output.status, delta="Approved" if output.status == "APPROVED" else "Max Retries")
    with col2:
        st.metric("Total Attempts", output.total_attempts)
    with col3:
        st.metric("Rejections Overcome", len(output.rejection_log))
    with col4:
        st.metric("Final Pass Rate", f"{int(output.final_evaluation.pass_rate * 100)}%")

    # Tabs
    tab_lesson, tab_rejection, tab_rubric, tab_memory = st.tabs([
        "📄 Passing Lesson",
        "🛑 Rejection Log & Diffs",
        "📋 Rubric Checkpoints Matrix",
        "🧠 Persistent Memory & Evolution"
    ])

    with tab_lesson:
        st.subheader("Final Polished Lesson (Ready to Ship)")
        st.download_button(
            label="📥 Download Lesson as Markdown",
            data=output.final_lesson,
            file_name="lesson_output.md",
            mime="text/markdown"
        )
        st.markdown(output.final_lesson)

    with tab_rejection:
        st.subheader("Rejection Audit Log")
        if not output.rejection_log:
            st.success("✨ Zero rejections! The lesson passed all 6 checkpoints on the very first attempt.")
        else:
            st.info(f"The system rejected {len(output.rejection_log)} draft(s) before meeting shipping criteria. Below is the full audit trail:")
            for entry in output.rejection_log:
                with st.container():
                    st.markdown(f"### ❌ Attempt #{entry.attempt} Rejection Record")
                    st.markdown(f"**Failed Checkpoints:** `{', '.join(entry.failed_checkpoints)}`")
                    st.markdown(f"**Evaluator Critique:** {entry.critique_summary}")
                    
                    st.markdown("**Detailed Diagnostic Failures:**")
                    for d in entry.detailed_failures:
                        st.markdown(f"- **{d['checkpoint']}**: {d['reason']}")
                        st.markdown(f"  * Evidence: `{d['evidence']}`")
                        st.markdown(f"  * Remediation: *{d['remediation']}*")

                    st.markdown("**Remediation Instructions Fed Into Next Prompt:**")
                    for rem in entry.remediation_plan:
                        st.markdown(f"- 🔧 {rem}")

                    st.markdown(f"**Subsequent Changes:** {entry.changes_made_in_next_draft}")
                    st.divider()

    with tab_rubric:
        st.subheader("Strict 6-Checkpoint Rubric (Binary Pass/Fail)")
        for chk in output.final_evaluation.checkpoints:
            css_class = "checkpoint-pass" if chk.passed else "checkpoint-fail"
            icon = "✅ PASS" if chk.passed else "❌ FAIL"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{chk.name} ({chk.checkpoint_id.value})</strong> — <strong>{icon}</strong><br/>
                <small><strong>Reason:</strong> {chk.reason}</small><br/>
                <small><strong>Evidence:</strong> {chk.evidence}</small>
            </div>
            """, unsafe_allow_html=True)

    with tab_memory:
        st.subheader("Persistent Cross-Run Memory & Evolved Rules")
        col_mem1, col_mem2 = st.columns(2)

        with col_mem1:
            st.markdown("#### 📈 Historical Checkpoint Failure Frequency")
            stats = memory.get_checkpoint_failure_frequencies()
            st.bar_chart(stats)

        with col_mem2:
            st.markdown("#### 🧠 Evolved Guidelines Sharpened from Failures")
            guidelines = memory.get_evolved_guidelines()
            if not guidelines:
                st.info("No guidelines evolved yet. Run a pipeline cycle to watch it learn!")
            else:
                for g in guidelines:
                    st.markdown(f"""
                    **{g.id}** ({g.target_checkpoint.value})
                    - **Rule:** {g.rule}
                    - **Anti-Pattern:** `{g.anti_pattern}`
                    - **Best Practice:** *{g.best_practice}*
                    - *Learned from topic: {g.learned_from_topic} (Triggered {g.frequency_triggered}x)*
                    """)
                    st.divider()

        st.markdown("#### 🗃️ Historical Runs Stored in Memory")
        all_runs = memory.get_all_runs()
        st.write(f"Total runs recorded in `data/memory/run_history.json`: **{len(all_runs)}**")
        if all_runs:
            run_data = [
                {
                    "Run ID": r.run_id,
                    "Timestamp": r.timestamp,
                    "Topic": r.topic,
                    "Status": r.status,
                    "Attempts": r.total_attempts,
                    "Rejections": r.rejection_count,
                    "Pass Rate": f"{int(r.final_pass_rate * 100)}%"
                }
                for r in reversed(all_runs[-10:])
            ]
            st.dataframe(run_data, use_container_width=True)

