#!/usr/bin/env python3
"""SENTINEL Week 4 End-to-End Test Script

Runs five representative queries through the Week 4 QC orchestrator.
"""

import json
from pathlib import Path

from week4_qc_orchestrator import SentinelWeek4Orchestrator

TEST_IMAGES = [
    {
        "image": "data/mvtec/bottle/test/broken_large/000.png",
        "question": "What is the defect, how severe is it, and which SOP should apply?"
    },
    {
        "image": "data/mvtec/bottle/test/broken_small/000.png",
        "question": "Is this bottle acceptable or should it be rejected?"
    },
    {
        "image": "data/mvtec/bottle/test/contamination/000.png",
        "question": "Describe the contamination issue and the safest action."
    },
    {
        "image": "data/mvtec/cable/test/cut_outer_insulation/000.png",
        "question": "Evaluate the cable insulation defect and recommend handling."
    },
    {
        "image": "data/mvtec/cable/test/bent_wire/000.png",
        "question": "Identify the wire defect and whether it is critical."
    }
]


def run_tests():
    orchestrator = SentinelWeek4Orchestrator()
    results = []

    for case in TEST_IMAGES:
        image_path = Path(case["image"])
        if not image_path.exists():
            print(f"Skipping missing test image: {image_path}")
            continue

        print(f"\n=== Running query for {image_path} ===")
        result = orchestrator.process_query(str(image_path), question=case["question"])
        results.append({
            "image": str(image_path),
            "question": case["question"],
            "reasoning": result.reasoning,
            "retrieval": result.retrieval,
            "metadata": result.metadata
        })
        print(json.dumps(result.reasoning, indent=2, ensure_ascii=False))

    output_path = Path("../data/week4_e2e_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved end-to-end test results to {output_path}")


if __name__ == "__main__":
    run_tests()
