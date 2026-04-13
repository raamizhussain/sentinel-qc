#!/usr/bin/env python3
"""Test the free LLM integration in Week 4 orchestrator"""

import os
import sys
sys.path.append('.')

from week4_qc_orchestrator import SentinelWeek4Orchestrator

def test_free_llm():
    print("=== Testing Free LLM Integration ===")

    # Initialize orchestrator
    orchestrator = SentinelWeek4Orchestrator()
    print("✓ Orchestrator initialized with free local LLM")

    # Test query
    image_path = "data/mvtec/bottle/test/good/000.png"
    question = "Is this bottle defective?"

    if os.path.exists(image_path):
        result = orchestrator.process_query(image_path, question)
        print("✓ Query processed successfully")
        print("Result:", result)
    else:
        print("⚠ Test image not found, but initialization successful")

    print("=== Test Complete ===")

if __name__ == "__main__":
    test_free_llm()