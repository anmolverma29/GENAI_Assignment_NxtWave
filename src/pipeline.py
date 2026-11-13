"""High-level pipeline facade and execution coordinator."""

import logging
from typing import Optional
from src.config import Config
from src.models import PipelineOutput
from src.graph import LessonPipelineGraph
from src.llm_provider import BaseLLMProvider, get_llm_provider
from src.memory import MemoryStore

logger = logging.getLogger(__name__)


def run_pipeline(
    topic: str = Config.DEFAULT_TOPIC,
    max_retries: int = Config.MAX_RETRIES,
    provider_type: Optional[str] = None,
    custom_llm: Optional[BaseLLMProvider] = None,
    memory_store: Optional[MemoryStore] = None
) -> PipelineOutput:
    """Executes the full agentic loop: Generate -> Evaluate -> Regenerate -> Evolve.
    
    Args:
        topic: The lesson topic to teach to complete beginners.
        max_retries: Max number of regeneration retries before hard termination (default: 2).
        provider_type: 'gemini', 'openai', or 'mock'.
        custom_llm: Optional pre-configured LLM provider instance.
        memory_store: Optional pre-configured MemoryStore instance.
        
    Returns:
        PipelineOutput with final lesson, rejection audit log, evaluations, and evolution trace.
    """
    llm = custom_llm or get_llm_provider(provider_type)
    memory = memory_store or MemoryStore()
    
    pipeline_graph = LessonPipelineGraph(
        llm_provider=llm,
        memory_store=memory
    )
    
    return pipeline_graph.run(topic=topic, max_retries=max_retries)

