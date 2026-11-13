"""Unified LLM Provider supporting Gemini, OpenAI, and Deterministic Mock simulation."""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel
from src.config import Config

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Generate unstructured text from a prompt."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        """Generate structured data adhering to a Pydantic schema."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider using google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model = model or Config.GEMINI_MODEL
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        full_contents = []
        if system_prompt:
            full_contents.append(f"System Instructions:\n{system_prompt}\n\n")
        full_contents.append(prompt)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents="".join(full_contents),
            config={"temperature": temperature}
        )
        return response.text or ""

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with a valid JSON object matching the JSON schema below.\n"
            f"Schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}\n\n"
            f"User Prompt:\n{prompt}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            }
        )
        raw_text = response.text or "{}"
        return response_model.model_validate_json(raw_text)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Using OpenAI beta parse for structured outputs
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_model,
            temperature=temperature
        )
        return completion.choices[0].message.parsed


class MockProvider(BaseLLMProvider):
    """Deterministic Mock Provider for reliable testing and offline evaluation.
    
    Simulates a realistic iterative refinement trajectory:
    - Attempt 1: Produces an imperfect draft with technical jargon and missing concrete example.
    - Attempt 2 (Retry 1): Receives critique and produces a high-quality, beginner-friendly lesson that passes all 6 checkpoints.
    """

    def __init__(self):
        self.generation_count = 0

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        self.generation_count += 1
        is_retry = "REMEDIATION FEEDBACK FROM PREVIOUS REJECTION" in prompt or "REJECTION CRITIQUE" in prompt or self.generation_count > 1

        if not is_retry:
            # Flawed Initial Draft (Attempt 1):
            # Intentionally fails on CHK_NO_JARGON and CHK_EXAMPLE
            return (
                "# Introduction to RAG (Retrieval-Augmented Generation)\n\n"
                "## What is RAG?\n"
                "RAG stands for Retrieval-Augmented Generation. It is an AI architecture that enhances "
                "large language models by dynamically retrieving relevant facts from an external vector index.\n\n"
                "## Why It Matters\n"
                "Standard LLMs often hallucinate or lack access to up-to-date data. With RAG, the model can query "
                "private enterprise stores rather than relying solely on frozen parametric weights.\n\n"
                "## How It Works\n"
                "The process works by converting documents into high-dimensional dense vector embeddings. "
                "These embeddings are indexed inside an approximate nearest neighbor HNSW vector database. "
                "At query time, the system performs a cosine similarity search across the vector space to retrieve "
                "top-k chunks, concatenates them into the LLM context window, and executes text completion.\n\n"
                "## Summary\n"
                "In conclusion, RAG combines retrieval algorithms with generative models to reduce hallucinations."
            )
        else:
            # Master Passing Draft (Attempt 2 - After Feedback):
            # Solves all failures: explains all jargon with metaphors, adds end-to-end concrete example, structured pedagogical flow
            return (
                "# 🧠 Introduction to RAG: Giving AI an Open-Book Exam\n\n"
                "Imagine you are taking a tough history exam. In a **closed-book exam**, you must rely entirely "
                "on what you memorized months ago. If you forgot a date or facts changed, you might guess or get it wrong.\n\n"
                "Now imagine an **open-book exam**. You have your textbook and notes right beside you. When a question is asked, "
                "you flip to the exact chapter, read the verified fact, and write a clear, accurate answer.\n\n"
                "That is the core idea behind **RAG (Retrieval-Augmented Generation)**. It transforms AI from a student taking a closed-book exam "
                "into a researcher taking an open-book exam.\n\n"
                "---\n\n"
                "## 1. What is RAG?\n\n"
                "**RAG** stands for **Retrieval-Augmented Generation**:\n"
                "- **Retrieval**: Finding the right reference information from an external source (like your company's documents, manuals, or articles).\n"
                "- **Augmented**: Adding those retrieved facts into the AI's question prompt as reference context.\n"
                "- **Generation**: Having the AI read that context and generate a natural, helpful, and accurate response.\n\n"
                "In plain words: **RAG is an AI technique that looks up verified facts in a library before answering your question.**\n\n"
                "---\n\n"
                "## 2. Why Does RAG Matter? (The Problem It Solves)\n\n"
                "Standard AI language models (like ChatGPT or Claude) are incredibly smart, but they suffer from three major limitations:\n\n"
                "1. **Hallucinations (Confident Guesswork)**: When an AI doesn't know something, it can invent plausible-sounding but false answers.\n"
                "2. **Outdated Knowledge (Knowledge Cutoff)**: Models only know what was in their training data. They don't know what happened yesterday or this morning.\n"
                "3. **No Private Company Knowledge**: A generic model knows about world history, but it has never seen your company's internal HR policies, customer support tickets, or confidential product roadmaps.\n\n"
                "**Why not just retrain the AI?**\n"
                "Retraining (or fine-tuning) a large model takes days, costs thousands of dollars, and must be repeated every time a document changes. "
                "**RAG solves this elegantly**: instead of changing the AI's brain, you simply hand it the right page to read on the fly.\n\n"
                "---\n\n"
                "## 3. How RAG Works (The 3-Step Journey)\n\n"
                "Think of RAG as a 3-step assembly line:\n\n"
                "```\n"
                "[Your Documents] ──► 1. Chunk & Index ──► 2. Retrieve Relevant Page ──► 3. Generate Grounded Answer\n"
                "```\n\n"
                "### Step 1: Ingestion & Indexing (Organizing the Library)\n"
                "- **Chunking**: Large documents (like a 200-page PDF) are cut into small, bite-sized index cards (paragraphs of 200–500 words).\n"
                "- **Embeddings (Numerical Meaning Fingerprints)**: Just like humans use keywords, computers convert text into a list of numbers called an *embedding*. Sentences with similar meanings get similar number patterns.\n"
                "- **Vector Database (The Smart Filing Cabinet)**: A specialized database that stores these chunks and their numerical fingerprints so we can find matching ideas in milliseconds.\n\n"
                "### Step 2: Retrieval (Finding the Relevant Index Cards)\n"
                "- When a user asks a question, the system turns the question into a numerical fingerprint.\n"
                "- It scans the vector database and retrieves the top 2 or 3 chunks that best match the question's meaning.\n\n"
                "### Step 3: Generation (Writing the Answer)\n"
                "- The system builds a new prompt for the AI:\n"
                "  > *\"You are a helpful assistant. Use ONLY the following reference notes to answer the user's question:\n"
                "  > [Retrieved Reference Chunks]\n"
                "  > User Question: ...\"*\n"
                "- The AI reads the provided notes and crafts an accurate, grounded reply.\n\n"
                "---\n\n"
                "## 4. Concrete Real-World Walkthrough\n\n"
                "Let's see RAG in action for **NovaTech Smartwatch Customer Support**:\n\n"
                "- **User Question**: *\"Can I wear my NovaWatch X in a heated sauna?\"*\n"
                "- **Without RAG**: The generic model might say: *\"Most smartwatches are water-resistant up to 50 meters, so you should be fine!\"* (Result: The user damages their watch because high heat ruins battery seals!).\n"
                "- **With RAG**:\n"
                "  1. **Retrieve**: The system searches NovaTech's 80-page user manual and retrieves Section 4.2:\n"
                "     > *\"Warning: The NovaWatch X is water-resistant up to 5 ATM, but MUST NOT be worn in steam rooms, saunas, or hot tubs exceeding 45°C as steam damages the protective acoustic membranes.\"*\n"
                "  2. **Augment**: The system constructs the prompt:\n"
                "     > *\"Reference Manual: [Section 4.2 Warning]\n"
                "     > Question: Can I wear my NovaWatch X in a heated sauna?\"*\n"
                "  3. **Generate**: The AI responds:\n"
                "     > *\"No, you should not wear your NovaWatch X in a sauna. According to Section 4.2 of the manual, steam and temperatures exceeding 45°C can damage the watch's protective seals.\"*\n\n"
                "The answer is accurate, safe, and directly cites company policy!\n\n"
                "---\n\n"
                "## 5. Key Takeaways & Summary\n\n"
                "- 📚 **RAG is an open-book exam for AI**: It retrieves real facts before generating an answer.\n"
                "- 🛡️ **Prevents hallucinations**: The AI answers based on verified documents, not guesswork.\n"
                "- ⚡ **Fresh & private data**: You can update your knowledge base instantly without expensive model retraining.\n"
                "- 🔄 **The 3 steps**: Chunk & Index (library) ➔ Retrieve (search) ➔ Generate (answer)."
            )

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        # In mock evaluation, analyze the prompt content to dynamically evaluate
        from src.models import RubricEvaluation, CheckpointResult, CheckpointID

        # If evaluating a draft:
        lesson_text = prompt
        
        # Check presence of jargon without explanation
        has_dense_jargon = "high-dimensional dense vector embeddings" in lesson_text and "fingerprint" not in lesson_text
        has_hnsw = "HNSW" in lesson_text and "filing cabinet" not in lesson_text
        has_cosine = "cosine similarity search" in lesson_text and "matching" not in lesson_text.lower()
        jargon_failed = has_dense_jargon or has_hnsw or has_cosine

        # Check concrete example
        has_concrete_example = ("NovaWatch" in lesson_text or "fictional" in lesson_text.lower() or 
                                ("user question" in lesson_text.lower() and "without rag" in lesson_text.lower()))

        # Check beginner friendly
        is_beginner = "open-book" in lesson_text.lower() or "exam" in lesson_text.lower() or "library" in lesson_text.lower()
        if not is_beginner and jargon_failed:
            beginner_passed = False
        else:
            beginner_passed = True

        # Check key points
        covers_what = "what is rag" in lesson_text.lower() or "retrieval-augmented generation" in lesson_text.lower()
        covers_why = "why" in lesson_text.lower() and ("hallucinat" in lesson_text.lower() or "problem" in lesson_text.lower())
        covers_how = "how" in lesson_text.lower() and ("step" in lesson_text.lower() or "retrieve" in lesson_text.lower())
        key_points_passed = covers_what and covers_why and covers_how

        # Check accuracy
        accurate_passed = "retrieval" in lesson_text.lower() and "fine-tuning" not in lesson_text.lower() or "instead of changing" in lesson_text.lower()

        # Check coherent flow
        coherent_passed = ("#" in lesson_text) and ("summary" in lesson_text.lower() or "takeaway" in lesson_text.lower())

        checkpoints = [
            CheckpointResult(
                checkpoint_id=CheckpointID.ACCURACY,
                name="Accurate & Grounded",
                passed=accurate_passed,
                reason="Concepts are accurately grounded in modern RAG architecture." if accurate_passed else "Contains factual errors or misleading definitions.",
                evidence="RAG correctly separates retrieval from generation." if accurate_passed else "Lacks factual clarity.",
                remediation="" if accurate_passed else "Ensure RAG is distinguished from fine-tuning."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.BEGINNER,
                name="Beginner-Friendly Language",
                passed=beginner_passed,
                reason="Uses intuitive open-book exam and library analogies accessible to complete novices." if beginner_passed else "Too academic and dry; assumes prior familiarity with AI terminology.",
                evidence="Analogy: 'open-book exam' used effectively." if beginner_passed else "Relies on technical terminology without framing.",
                remediation="" if beginner_passed else "Introduce the open-book exam metaphor early to ground intuition."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.EXAMPLE,
                name="Teaches by Concrete Example",
                passed=has_concrete_example,
                reason="Includes a full walkthrough with a customer support query, retrieved document snippet, and comparison." if has_concrete_example else "Lacks an end-to-end real-world walkthrough demonstrating query, retrieval, and generation.",
                evidence="Walkthrough: NovaWatch X heated sauna support inquiry." if has_concrete_example else "No concrete user inquiry with specific retrieved text provided.",
                remediation="" if has_concrete_example else "Add a realistic scenario showing: (1) User asks a question, (2) System retrieves specific chunk, (3) Model answers using that chunk."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.NO_JARGON,
                name="Clear, No Unexplained Jargon",
                passed=not jargon_failed,
                reason="All technical terms (embeddings, chunking, vector database) are defined before or alongside use." if not jargon_failed else "Introduced technical jargon (dense embeddings, HNSW, cosine similarity) without defining them for a beginner.",
                evidence="Embeddings defined as 'numerical meaning fingerprints', chunking as 'index cards'." if not jargon_failed else "Found: 'high-dimensional dense vector embeddings', 'HNSW vector database' with no layman explanation.",
                remediation="" if not jargon_failed else "Define embeddings using the 'numerical fingerprint' analogy and vector DB as 'smart filing cabinet'."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.KEY_POINTS,
                name="Covers All Key Pillars",
                passed=key_points_passed,
                reason="Thoroughly covers What it is, Why it matters (hallucinations/freshness/private data), and How it works (3-step pipeline)." if key_points_passed else "Fails to cover all three pillars adequately.",
                evidence="Sections 1, 2, and 3 explicitly cover What, Why, and How." if key_points_passed else "Missing clear coverage of all three pillars.",
                remediation="" if key_points_passed else "Ensure distinct sections for What it is, Why it matters, and How it works."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.COHERENT_FLOW,
                name="Coherent Pedagogical Flow",
                passed=coherent_passed,
                reason="Clear hierarchical progression: Hook -> Definition -> Problem -> Steps -> Example -> Key Takeaways." if coherent_passed else "Teaching flow is disjointed or lacks structure.",
                evidence="Markdown headings structure the learning path from simple to practical." if coherent_passed else "Needs better structural progression.",
                remediation="" if coherent_passed else "Organize with clear numbered section headers and a concluding summary."
            )
        ]

        failed = [c.checkpoint_id for c in checkpoints if not c.passed]
        all_passed = len(failed) == 0
        pass_rate = (len(checkpoints) - len(failed)) / len(checkpoints)

        overall_feedback = (
            "Congratulations! The lesson clears all 6 hard checkpoints and is ready to ship."
            if all_passed else
            f"The lesson failed {len(failed)} critical checkpoints: {', '.join(failed)}. Must address specific remediation feedback."
        )

        return RubricEvaluation(
            checkpoints=checkpoints,
            all_passed=all_passed,
            failed_checkpoints=failed,
            overall_feedback=overall_feedback,
            pass_rate=pass_rate
        )


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """Factory to get the appropriate LLM provider."""
    provider_name = (provider_type or Config.get_effective_provider()).lower()
    
    if provider_name == "gemini":
        try:
            return GeminiProvider()
        except Exception as e:
            logger.warning(f"Could not initialize GeminiProvider ({e}); falling back to MockProvider.")
            return MockProvider()
    elif provider_name == "openai":
        try:
            return OpenAIProvider()
        except Exception as e:
            logger.warning(f"Could not initialize OpenAIProvider ({e}); falling back to MockProvider.")
            return MockProvider()
    else:
        return MockProvider()

