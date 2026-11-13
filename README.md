# 🏗️ Self-Evaluating Lesson Content Generator

> An agentic educational content generator that autonomously drafts beginner lessons, rigorously evaluates them against a strict binary pass/fail rubric, and regenerates them based on evaluator feedback until they clear every checkpoint before human review.

**Topic for Submission:** `Introduction to RAG (Retrieval-Augmented Generation)`  
**Technology Stack:** Python 3.13, LangGraph, LangChain, Pydantic v2, Streamlit, Rich, Pytest.

---
## 🎯 Assignment Deliverables
1. **GitHub Repository:** https://github.com/anmolverma29/GENAI_Assignment_NxtWave
2. **Final Lesson Document:** https://docs.google.com/document/d/1h7n3lNc96TpcQu851MHtc9p9KXMLNgl3DemMxy1KwdA/edit?usp=sharing
3. **Loom Video Walkthrough:** [Link to Loom Video] (15-20 min demo showing the end-to-end loop and catching a deliberate error).

## 🌟 Key Highlights

- **🔁 Full Generate ➔ Evaluate ➔ Regenerate Loop:**
  Autonomous agentic cycle orchestrated with **LangGraph**. If any checkpoint fails, the evaluator constructs a diagnostic remediation packet and feeds it back into the generator for revision.
- **📋 6 Hard Pass/Fail Checkpoints (No Partial Credit):**
  - `accurate & grounded`
  - `beginner-friendly language`
  - `teaches by example`
  - `clear, no unexplained jargon`
  - `covers the key points` (What, Why, How)
  - `coherent teaching flow`
- **🛑 Transparent Rejection Audit Log:**
  Captures the exact failure reasons, text evidence, critique, and what changed on retry.
- **🧠 Cross-Run Memory & Self-Evolution:**
  Persists historical run data across runs (`run_history.json`). A meta-optimizer extracts anti-patterns from rejections and stores sharpened guidelines (`evolved_guidelines.json`) that are automatically injected into subsequent generation prompts.
- **⚡ Dual-Layer Evaluator:**
  Combines deterministic heuristic verification (checking jargon, examples, and pillar structure) with semantic LLM evaluation to eliminate evaluator leniency and score drift.
- **🖥️ Multiple Interfaces:**
  - **CLI Tool** (`main.py`): Formatted terminal interface with color-coded tables and Markdown viewer.
  - **Interactive Web App** (`app.py`): Streamlit dashboard with LangGraph DAG visualizer, rejection audit inspector, side-by-side diffs, and memory metrics.
  - **Automated Test Suite** (`pytest`): 13 unit and integration tests verifying all rubric checkpoints, feedback loops, bounded termination, and persistence.
- **🔌 Multi-Provider Support:**
  Supports **Google Gemini** (`gemini-2.5-flash`), **OpenAI** (`gpt-4o-mini`), and a deterministic **Mock/Simulation Engine** that runs offline with zero setup!

---

## 📁 Repository Structure

```
GENAI_Assignment_NxtWave/
├── README.md                      # Comprehensive guide and documentation
├── ARCHITECTURE.md                # In-depth architectural design, trade-offs, and state machine
├── requirements.txt               # Dependencies
├── .env.example                   # Environment configuration template
├── app.py                         # Interactive Streamlit Web UI
├── main.py                        # Command Line Interface
├── data/
│   ├── memory/                    # Persistent cross-run storage
│   │   ├── run_history.json       # Historical telemetry of runs, attempts, and pass rates
│   │   └── evolved_guidelines.json# Self-evolved prompt rules and anti-patterns
│   └── outputs/                   # Shipped submission artifacts
│       ├── rag_lesson_final.md    # Final passing lesson for "Introduction to RAG"
│       └── rag_rejection_log.json # Rejection audit log for the submission run
├── src/
│   ├── models.py                  # Pydantic data schemas & checkpoint enums
│   ├── config.py                  # Environment and configuration settings
│   ├── llm_provider.py            # Gemini, OpenAI, and Deterministic Mock provider
│   ├── prompts.py                 # System, evaluation, revision, and evolution prompts
│   ├── evaluator.py               # 6-checkpoint dual-layer rubric evaluator
│   ├── generator.py               # Lesson generator & feedback-conditioned reviser
│   ├── memory.py                  # Cross-run persistence manager
│   ├── evolution.py               # Self-evolution meta-optimizer
│   ├── graph.py                   # LangGraph StateGraph orchestration
│   └── pipeline.py                # High-level pipeline entrypoint
└── tests/
    ├── test_rubric.py             # Tests for each of the 6 hard checkpoints
    ├── test_memory.py             # Tests for JSON persistence and frequency metrics
    ├── test_evolution.py          # Tests for guideline evolution from rejection logs
    └── test_graph.py              # Tests for LangGraph cycle and bounded termination
```

---

## 🚀 Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
If you wish to use live LLM APIs, copy `.env.example` to `.env` and provide your API keys:
```bash
copy .env.example .env
```
*(Note: If no API keys are provided, the system automatically runs using the built-in deterministic simulation engine, demonstrating the complete generate ➔ evaluate ➔ reject ➔ regenerate loop out of the box).*

---

## 💻 Running the Application

### Option A: Command Line Interface (CLI)
Run the pipeline directly from your terminal:
```bash
python main.py --topic "Introduction to RAG"
```

To run with a live provider (e.g. Gemini):
```bash
python main.py --topic "Introduction to RAG" --provider gemini
```

### Option B: Interactive Streamlit Web App
Launch the rich visual dashboard:
```bash
streamlit run app.py
```
**Features of the Web App:**
- 🗺️ Visual LangGraph State Machine diagram.
- 📄 Real-time rendered passing lesson with 1-click Markdown download.
- 🛑 Detailed rejection audit log showing failed checkpoints, evidence, and remediations.
- 📋 Visual 6-checkpoint pass/fail matrix.
- 🧠 Historical failure frequency charts and persistent evolved guidelines viewer.

---

## 🧪 Running the Test Suite

Execute the 13 automated test suites:
```bash
python -m pytest tests/ -v
```

All 13 tests verify:
- Rubric checks catch ungrounded jargon and missing concrete examples.
- Binary evaluation enforces strict pass/fail with no partial credit.
- Feedback is correctly synthesized into prescriptive remediation plans.
- Bounded termination guarantees that the cycle halts at `max_retries`.
- Persistent memory stores runs and increments failure frequencies.
- Evolved rules are correctly synthesized from rejection logs.

---

## 📦 Submission Artifacts

The final generated submission outputs are located in:
1. **Passing Lesson:** [`data/outputs/rag_lesson_final.md`]
   - Opens with the relatable **Open-Book vs Closed-Book Exam** analogy.
   - Covers **What it is**, **Why it matters** (hallucinations, fresh data, enterprise privacy), and **How it works** (Ingest ➔ Retrieve ➔ Generate).
   - Features an end-to-end customer support walkthrough (**NovaTech Smartwatch Sauna inquiry**).
   - Translates all technical terms (*embeddings = numerical fingerprints*, *vector DB = smart filing cabinet*, *chunking = index cards*).
2. **Rejection Audit Log:** [`data/outputs/rag_rejection_log.json`]
   - Documents Attempt 1's rejection on `CHK_BEGINNER`, `CHK_EXAMPLE`, and `CHK_NO_JARGON`.
   - Records the diagnostic evidence, evaluator critique, and remediation steps that steered Attempt 2 to a 100% pass!
3. **Architecture Document:** [`ARCHITECTURE.md`]
   - Explains state machines, rubric trade-offs, dual-layer evaluation, and self-evolution mechanisms.

