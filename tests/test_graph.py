"""Integration tests for the LangGraph agentic loop and bounded termination."""

import pytest
from pathlib import Path
from src.config import Config
from src.models import PipelineOutput
from src.graph import LessonPipelineGraph
from src.memory import MemoryStore
from src.llm_provider import MockProvider


@pytest.fixture
def isolated_graph(tmp_path):
    """Provides a pipeline graph isolated in a temp directory."""
    history_file = tmp_path / "test_history.json"
    guidelines_file = tmp_path / "test_guidelines.json"
    memory = MemoryStore(history_path=history_file, guidelines_path=guidelines_file)
    mock_llm = MockProvider()
    return LessonPipelineGraph(llm_provider=mock_llm, memory_store=memory), memory


def test_langgraph_full_loop_success(isolated_graph):
    """Verify complete agentic loop: Generate -> Evaluate -> Reject -> Regenerate -> Approve."""
    graph, memory = isolated_graph
    
    output = graph.run(topic="Introduction to RAG", max_retries=2)
    
    # Assertions on pipeline output
    assert output.status == "APPROVED"
    assert output.total_attempts == 2  # Attempt 1 failed, Attempt 2 passed
    assert len(output.rejection_log) == 1
    
    # Verify rejection record
    rejection = output.rejection_log[0]
    assert rejection.attempt == 1
    assert "Teaches by Concrete Example" in rejection.failed_checkpoints or "CHK_EXAMPLE" in rejection.failed_checkpoints or len(rejection.failed_checkpoints) > 0
    assert len(rejection.remediation_plan) > 0

    # Verify final lesson passes all checkpoints
    assert output.final_evaluation.all_passed
    assert output.final_evaluation.pass_rate == 1.0

    # Verify memory was updated
    runs = memory.get_all_runs()
    assert len(runs) == 1
    assert runs[0].status == "APPROVED"

    # Verify evolved guidelines were generated
    guidelines = memory.get_evolved_guidelines()
    assert len(guidelines) > 0


def test_langgraph_bounded_termination(tmp_path):
    """Verify that loop strictly terminates and does not loop infinitely if retries are capped at 1."""
    # Create a stub LLM that always returns flawed content
    class AlwaysFlawedLLM(MockProvider):
        def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
            return "RAG uses dense vectors and HNSW."  # Always fails CHK_NO_JARGON and CHK_EXAMPLE

    history_file = tmp_path / "test_history.json"
    guidelines_file = tmp_path / "test_guidelines.json"
    memory = MemoryStore(history_path=history_file, guidelines_path=guidelines_file)
    
    graph = LessonPipelineGraph(llm_provider=AlwaysFlawedLLM(), memory_store=memory)
    
    # Run with max_retries = 1
    output = graph.run(topic="Introduction to RAG", max_retries=1)

    # Must terminate with MAX_RETRIES_REACHED rather than looping infinitely
    assert output.status == "MAX_RETRIES_REACHED"
    assert len(output.rejection_log) == 2  # Initial attempt + 1 retry = 2 rejections
    assert not output.final_evaluation.all_passed

