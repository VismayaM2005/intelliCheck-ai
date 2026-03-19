import sys
import os
import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelliCheck_ai import clean_text, get_sentences, plagiarism_check
from rag_engine import RAGEngine
from enhanced_paraphraser import paraphrase_advanced

# ---------------------------------------------------------
# Feature 8: Automated Testing Framework
# 1. Unit Testing for Preprocessing Pipeline
# ---------------------------------------------------------
def test_clean_text():
    raw = "This   is \n a messy... text with \t tabs!"
    cleaned = clean_text(raw)
    assert cleaned == "This is a messy. text with tabs!"
    assert "\n" not in cleaned
    assert "  " not in cleaned

def test_get_sentences():
    text = "This is the first sentence. And here is another long enough sentence."
    sents = get_sentences(text)
    assert len(sents) == 2
    assert "This is the first sentence." in sents

# ---------------------------------------------------------
# Feature 8: Automated Testing Framework
# 2. Integration Testing for RAG Pipeline
# ---------------------------------------------------------
@pytest.fixture
def sample_rag_engine():
    engine = RAGEngine()
    docs = [
        "Machine learning is a field of artificial intelligence.",
        "Generative AI refers to algorithms that can create new content.",
        "Docker is a set of platform as a service products used for containerization."
    ]
    engine.build_index(docs)
    return engine

def test_rag_retrieval(sample_rag_engine):
    query = "What is artificial intelligence?"
    D, I = sample_rag_engine.search(query, k=1)
    # The search should return the most relevant document at index 0
    assert I[0][0] == 0

# ---------------------------------------------------------
# Feature 8: Automated Testing Framework
# 3. Model Output Validation Tests
# ---------------------------------------------------------
def test_model_output_validation():
    user_text = "Generative AI creates novel text based on training data."
    # Since calling the actual LLM might be slow and require API, 
    # we test the structure of the output or mock the response
    variants = paraphrase_advanced(
        user_text,
        num_variants=1,
        strategy="zero_shot",
        temperature=0.7,
        top_p=0.9,
        top_k=50 
    )
    assert isinstance(variants, list)
    if variants:
        assert "text" in variants[0]
        assert "strategy" in variants[0]
        assert "risk_score" in variants[0]
