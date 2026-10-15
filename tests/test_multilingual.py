"""
Tests for Multilingual RAG features in ragspark.
"""

from __future__ import annotations

import pytest
from ragspark import detect_language
from ragspark.prompt_eng import build, build_law
from ragspark.rag import create, ask


def test_detect_language():
    # Hindi (Devanagari)
    assert detect_language("कंपनी का नियम क्या है?") == "Hindi"
    
    # Spanish (stop words)
    assert detect_language("¿Cuál es la política de reembolso de la empresa?") == "Spanish"
    
    # French
    assert detect_language("Quelle est la politique de retour de l'entreprise?") == "French"
    
    # German
    assert detect_language("Was ist die Rückerstattungsrichtlinie des Unternehmens?") == "German"
    
    # Chinese
    assert detect_language("退款政策是什么？") == "Chinese"
    
    # Russian (Cyrillic)
    assert detect_language("Каковы правила возврата денег?") == "Russian"
    
    # Arabic
    assert detect_language("ما هي سياسة الاسترجاع؟") == "Arabic"
    
    # English fallback
    assert detect_language("What is the company refund policy?") == "English"


def test_prompt_engineering_multilingual_directive():
    built = build(
        query="What is the refund window?",
        context_items=["Customers can return items within 30 days for a full refund."],
        language="Spanish",
    )
    assert built.language == "Spanish"
    assert "Output Language: Spanish" in built.user_prompt
    assert "strictly and fluently in **Spanish**" in built.user_prompt


def test_prompt_engineering_multilingual_auto():
    built = build(
        query="क्या ग्राहक 30 दिनों के बाद सामान वापस कर सकता है?",
        context_items=["Customers can return items within 30 days for a full refund."],
        language="auto",
    )
    assert built.language == "Hindi"
    assert "Output Language: Hindi" in built.user_prompt


def test_rag_pipeline_multilingual():
    doc_text = """
    Enterprise Policy 2026:
    Section 1. Return Window.
    All customers are eligible for a complete refund within 30 days of shipment receipt.
    Section 2. Hardware Replacement.
    Damaged hardware will be replaced within 48 hours under the premium warranty plan.
    """
    
    bot = create(
        source=doc_text,
        vectordb="memory",
        llm_provider="extractive",
        language="Hindi",
    )
    
    res = bot.ask("What is the return window?", language="Hindi")
    assert res.language == "Hindi"
    assert "Hindi" in res.built_prompt.user_prompt
    assert len(res.sources) > 0


def test_top_level_ask_multilingual():
    doc_text = "Standard Operating Procedure: All servers must restart at 02:00 UTC every Sunday."
    res = ask(
        source=doc_text,
        query="When do servers restart?",
        vectordb="memory",
        language="German",
    )
    assert res.language == "German"
    assert "German" in res.built_prompt.user_prompt
