"""Unit tests for the 6 hard pass/fail rubric checkpoints and evaluator."""

import pytest
from src.models import CheckpointID, RubricEvaluation
from src.evaluator import LessonEvaluator, HeuristicVerifier
from src.llm_provider import MockProvider


def test_heuristic_verifier_jargon_failure():
    """Verify that unexplained technical jargon is caught and failed."""
    bad_text = "RAG works by indexing documents using high-dimensional dense vector embeddings and HNSW graphs."
    passed, reason, evidence = HeuristicVerifier.check_jargon(bad_text)
    assert not passed
    assert "embeddings" in reason or "dense vector" in reason


def test_heuristic_verifier_jargon_pass():
    """Verify that technical jargon accompanied by intuitive analogies passes."""
    good_text = (
        "The system creates embeddings, which are numerical fingerprints representing the meaning of text. "
        "It stores them in a vector database, like a smart filing cabinet, after chunking documents into small index cards."
    )
    passed, reason, evidence = HeuristicVerifier.check_jargon(good_text)
    assert passed


def test_heuristic_verifier_example_failure():
    """Verify that lessons lacking concrete query-retrieval-answer walkthroughs fail."""
    abstract_text = "RAG allows an AI to retrieve documents from a database and answer questions accurately."
    passed, reason, evidence = HeuristicVerifier.check_example(abstract_text)
    assert not passed


def test_heuristic_verifier_example_pass():
    """Verify that lessons containing an explicit end-to-end scenario pass."""
    concrete_text = (
        "User Question: Can I wear this watch in a sauna?\n"
        "Retrieved Section 4: Do not wear in saunas exceeding 45C.\n"
        "Generated Answer: No, sauna heat damages the water seal."
    )
    passed, reason, evidence = HeuristicVerifier.check_example(concrete_text)
    assert passed


def test_heuristic_verifier_key_points():
    """Verify What, Why, and How detection."""
    incomplete_text = "What is RAG? It is Retrieval-Augmented Generation. How it works: it retrieves facts."
    passed, reason, remediation = HeuristicVerifier.check_key_points(incomplete_text)
    assert not passed  # Missing "Why it matters"

    complete_text = (
        "## 1. What is RAG?\nIt is an open-book exam technique.\n"
        "## 2. Why RAG Matters\nIt solves LLM hallucinations and outdated data.\n"
        "## 3. How RAG Works\nStep 1: Chunk, Step 2: Retrieve, Step 3: Generate."
    )
    passed, reason, _ = HeuristicVerifier.check_key_points(complete_text)
    assert passed


def test_evaluator_binary_no_partial_credit():
    """Verify that if even 1 checkpoint fails, all_passed is strictly False."""
    evaluator = LessonEvaluator(llm_provider=MockProvider())
    
    # Flawed text (from Attempt 1)
    flawed_text = (
        "# Introduction to RAG\n"
        "## What is RAG?\nRetrieval-Augmented Generation.\n"
        "## Why It Matters\nSolves hallucinations.\n"
        "## How It Works\nUses high-dimensional dense vector embeddings and HNSW cosine similarity search."
    )
    result = evaluator.evaluate(flawed_text, topic="Introduction to RAG")
    
    assert not result.all_passed
    assert len(result.failed_checkpoints) > 0
    assert CheckpointID.NO_JARGON in result.failed_checkpoints or CheckpointID.EXAMPLE in result.failed_checkpoints
    assert result.pass_rate < 1.0


def test_evaluator_master_draft_passes_all():
    """Verify that a comprehensive, beginner-friendly lesson passes all 6 checkpoints."""
    mock = MockProvider()
    evaluator = LessonEvaluator(llm_provider=mock)
    
    # Fetch the retry/passing lesson
    passing_lesson = mock.generate_text("Prompt with REMEDIATION FEEDBACK FROM PREVIOUS REJECTION")
    result = evaluator.evaluate(passing_lesson, topic="Introduction to RAG")
    
    assert result.all_passed
    assert len(result.failed_checkpoints) == 0
    assert result.pass_rate == 1.0
    for chk in result.checkpoints:
        assert chk.passed

