"""LangGraph StateGraph orchestration for the Self-Evaluating Lesson Content Generator."""

import datetime
import logging
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from src.config import Config
from src.models import (
    CheckpointID,
    CheckpointResult,
    RubricEvaluation,
    RejectionLogEntry,
    RunMemoryRecord,
    PipelineOutput
)
from src.evaluator import LessonEvaluator
from src.generator import LessonGenerator
from src.memory import MemoryStore
from src.evolution import EvolutionOptimizer
from src.llm_provider import BaseLLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


class LessonGraphState(TypedDict):
    """The complete state container flowing through the LangGraph agentic loop."""
    topic: str
    target_audience: str
    attempt: int
    max_retries: int
    current_lesson: str
    previous_draft: Optional[str]
    latest_evaluation: Optional[Dict[str, Any]]
    rejection_log: List[Dict[str, Any]]
    is_approved: bool
    status: str
    evolved_guidelines: List[str]
    execution_trace: List[Dict[str, Any]]


class LessonPipelineGraph:
    """Builder and executor for the LangGraph agentic loop."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        memory_store: Optional[MemoryStore] = None
    ):
        self.llm = llm_provider or get_llm_provider()
        self.memory = memory_store or MemoryStore()
        self.generator = LessonGenerator(llm_provider=self.llm)
        self.evaluator = LessonEvaluator(llm_provider=self.llm)
        self.evolution = EvolutionOptimizer(memory_store=self.memory, llm_provider=self.llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        """Builds and compiles the cyclic LangGraph workflow."""
        workflow = StateGraph(LessonGraphState)

        # Add Nodes
        workflow.add_node("initialize", self._node_initialize)
        workflow.add_node("generate", self._node_generate)
        workflow.add_node("evaluate", self._node_evaluate)
        workflow.add_node("record_rejection", self._node_record_rejection)
        workflow.add_node("evolve", self._node_evolve)
        workflow.add_node("finalize", self._node_finalize)

        # Set Entrypoint
        workflow.set_entry_point("initialize")

        # Define Edges
        workflow.add_edge("initialize", "generate")
        workflow.add_edge("generate", "evaluate")

        # Conditional Edge after Evaluation
        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "approved": "evolve",
                "rejected": "record_rejection",
            }
        )

        # Conditional Edge after Recording Rejection
        workflow.add_conditional_edges(
            "record_rejection",
            self._route_after_rejection,
            {
                "retry": "generate",
                "max_retries_exceeded": "evolve",
            }
        )

        workflow.add_edge("evolve", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    # Node Implementations
    def _node_initialize(self, state: LessonGraphState) -> Dict[str, Any]:
        """Loads persistent memory and sets up initial run configuration."""
        active_guidelines = self.memory.get_active_guidelines_text()
        trace_entry = {
            "node": "initialize",
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"Initialized run for topic '{state['topic']}' with {len(active_guidelines)} evolved guidelines."
        }
        return {
            "attempt": 1,
            "rejection_log": [],
            "evolved_guidelines": active_guidelines,
            "status": "INITIALIZED",
            "execution_trace": [trace_entry]
        }

    def _node_generate(self, state: LessonGraphState) -> Dict[str, Any]:
        """Generates initial draft or regenerates based on feedback."""
        attempt = state["attempt"]
        topic = state["topic"]
        guidelines = state.get("evolved_guidelines", [])

        if attempt == 1:
            lesson_text = self.generator.generate_initial_draft(
                topic=topic,
                evolved_guidelines=guidelines
            )
            action_desc = f"Generated initial draft (Attempt #{attempt})."
        else:
            # Reconstruct RubricEvaluation from previous state
            eval_dict = state["latest_evaluation"] or {}
            eval_obj = RubricEvaluation.model_validate(eval_dict)
            lesson_text = self.generator.revise_draft(
                topic=topic,
                previous_draft=state.get("current_lesson", ""),
                evaluation=eval_obj,
                attempt=attempt,
                evolved_guidelines=guidelines
            )
            action_desc = f"Regenerated draft (Attempt #{attempt}) addressing rejection critique."

        trace_entry = {
            "node": "generate",
            "attempt": attempt,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": action_desc,
            "lesson_preview": lesson_text[:180] + "..."
        }

        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "previous_draft": state.get("current_lesson"),
            "current_lesson": lesson_text,
            "status": f"GENERATED_ATTEMPT_{attempt}",
            "execution_trace": trace
        }

    def _node_evaluate(self, state: LessonGraphState) -> Dict[str, Any]:
        """Applies the strict 6-checkpoint rubric to the current draft."""
        topic = state["topic"]
        lesson_text = state["current_lesson"]
        evaluation = self.evaluator.evaluate(lesson_text=lesson_text, topic=topic)

        trace_entry = {
            "node": "evaluate",
            "attempt": state["attempt"],
            "timestamp": datetime.datetime.now().isoformat(),
            "all_passed": evaluation.all_passed,
            "pass_rate": evaluation.pass_rate,
            "failed_checkpoints": [c.value for c in evaluation.failed_checkpoints],
            "feedback": evaluation.overall_feedback
        }

        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "latest_evaluation": evaluation.model_dump(),
            "is_approved": evaluation.all_passed,
            "status": "APPROVED" if evaluation.all_passed else "EVALUATION_FAILED",
            "execution_trace": trace
        }

    def _node_record_rejection(self, state: LessonGraphState) -> Dict[str, Any]:
        """Logs rejection reasons, evidence, and prescriptive remediation plan."""
        attempt = state["attempt"]
        eval_dict = state["latest_evaluation"] or {}
        evaluation = RubricEvaluation.model_validate(eval_dict)

        failed_ids = [chk.checkpoint_id.value for chk in evaluation.checkpoints if not chk.passed]
        failed_names = [chk.name for chk in evaluation.checkpoints if not chk.passed]
        detailed_failures = [
            {
                "checkpoint": chk.name,
                "id": chk.checkpoint_id.value,
                "reason": chk.reason,
                "evidence": chk.evidence,
                "remediation": chk.remediation
            }
            for chk in evaluation.checkpoints if not chk.passed
        ]

        remediation_plan = [
            f"Fix {chk.name}: {chk.remediation}"
            for chk in evaluation.checkpoints if not chk.passed and chk.remediation
        ]

        rejection_entry = RejectionLogEntry(
            attempt=attempt,
            timestamp=datetime.datetime.now().isoformat(),
            failed_checkpoints=failed_ids,
            critique_summary=evaluation.overall_feedback,
            detailed_failures=detailed_failures,
            remediation_plan=remediation_plan,
            changes_made_in_next_draft=f"Revision will address: {', '.join(failed_names)}"
        )

        rejection_log = list(state.get("rejection_log", []))
        rejection_log.append(rejection_entry.model_dump())

        trace_entry = {
            "node": "record_rejection",
            "attempt": attempt,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"Recorded rejection for Attempt #{attempt}. Failed: {', '.join(failed_names)}",
            "remediation_plan": remediation_plan
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        # Increment attempt counter for next cycle
        return {
            "attempt": attempt + 1,
            "rejection_log": rejection_log,
            "status": f"REJECTED_ATTEMPT_{attempt}",
            "execution_trace": trace
        }

    def _node_evolve(self, state: LessonGraphState) -> Dict[str, Any]:
        """Learns from failures in this run and updates persistent guidelines."""
        raw_rejections = state.get("rejection_log", [])
        rejection_objects = [RejectionLogEntry.model_validate(r) for r in raw_rejections]

        evolved = self.evolution.evolve_from_rejection_log(
            topic=state["topic"],
            rejection_log=rejection_objects
        )

        trace_entry = {
            "node": "evolve",
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"Extracted and saved {len(evolved)} new/updated guidelines to memory."
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "evolved_guidelines": self.memory.get_active_guidelines_text(),
            "execution_trace": trace
        }

    def _node_finalize(self, state: LessonGraphState) -> Dict[str, Any]:
        """Persists final lesson, rejection logs, and run memory to disk."""
        eval_dict = state["latest_evaluation"] or {}
        evaluation = RubricEvaluation.model_validate(eval_dict) if eval_dict else None

        raw_rejections = state.get("rejection_log", [])
        rejection_objects = [RejectionLogEntry.model_validate(r) for r in raw_rejections]

        is_approved = state["is_approved"]
        final_status = "APPROVED" if is_approved else "MAX_RETRIES_REACHED"
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Persist run to memory store
        record = RunMemoryRecord(
            run_id=run_id,
            timestamp=datetime.datetime.now().isoformat(),
            topic=state["topic"],
            status=final_status,
            total_attempts=state["attempt"] if is_approved else state["attempt"] - 1,
            final_pass_rate=evaluation.pass_rate if evaluation else 0.0,
            rejection_count=len(rejection_objects),
            rejection_log=rejection_objects,
            final_lesson=state["current_lesson"]
        )
        self.memory.save_run(record)

        # Write output files to disk
        Config.FINAL_LESSON_OUTPUT_PATH.write_text(state["current_lesson"], encoding="utf-8")
        
        import json
        with open(Config.REJECTION_LOG_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in rejection_objects], f, indent=2, ensure_ascii=False)

        trace_entry = {
            "node": "finalize",
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"Pipeline complete. Status: {final_status}. Outputs written to data/outputs."
        }
        trace = list(state.get("execution_trace", []))
        trace.append(trace_entry)

        return {
            "status": final_status,
            "execution_trace": trace
        }

    # Conditional Routing Logic
    def _route_after_evaluation(self, state: LessonGraphState) -> str:
        """Route to approval flow or rejection handler."""
        if state["is_approved"]:
            return "approved"
        return "rejected"

    def _route_after_rejection(self, state: LessonGraphState) -> str:
        """Determines whether to retry or terminate (max retries reached)."""
        attempt = state["attempt"]  # Note: already incremented in record_rejection
        max_allowed_attempts = state["max_retries"] + 1  # 1 initial + max_retries
        
        if attempt <= max_allowed_attempts:
            return "retry"
        else:
            return "max_retries_exceeded"

    def run(self, topic: str = Config.DEFAULT_TOPIC, max_retries: int = Config.MAX_RETRIES) -> PipelineOutput:
        """Executes the full compiled LangGraph workflow."""
        initial_state: LessonGraphState = {
            "topic": topic,
            "target_audience": "Beginner with zero technical background",
            "attempt": 1,
            "max_retries": max_retries,
            "current_lesson": "",
            "previous_draft": None,
            "latest_evaluation": None,
            "rejection_log": [],
            "is_approved": False,
            "status": "NOT_STARTED",
            "evolved_guidelines": [],
            "execution_trace": []
        }

        final_state = self.graph.invoke(initial_state)

        eval_dict = final_state["latest_evaluation"] or {}
        evaluation = RubricEvaluation.model_validate(eval_dict)
        raw_rejections = final_state.get("rejection_log", [])
        rejection_objects = [RejectionLogEntry.model_validate(r) for r in raw_rejections]

        return PipelineOutput(
            topic=final_state["topic"],
            status=final_state["status"],
            total_attempts=len(rejection_objects) + (1 if final_state["is_approved"] else 0),
            final_lesson=final_state["current_lesson"],
            rejection_log=rejection_objects,
            final_evaluation=evaluation,
            evolved_guidelines_used=final_state.get("evolved_guidelines", []),
            metadata={
                "execution_trace": final_state.get("execution_trace", []),
                "max_retries": max_retries
            }
        )
