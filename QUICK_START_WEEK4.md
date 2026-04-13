# SENTINEL Week 4 - Quick Start Guide

## LLM Reasoning (Free Local Model)

The system now uses a free local LLM model (google/flan-t5-small) that runs on your CPU without any API keys or costs.

No setup required - LLM reasoning is automatically enabled.

---

## Run a Query

### Basic Query (No LLM)
```bash
cd scripts
python -c "
import sys
sys.path.insert(0, '.')
from week4_qc_orchestrator import SentinelWeek4Orchestrator

orch = SentinelWeek4Orchestrator(yolo_weights=None)
result = orch.process_query(
    image_path='../data/mvtec/bottle/test/good/000.png',
    question='Assess quality and identify defects'
)
print(f'Status: {result.reasoning}')
"
```

### Via Command Line (With YOLO Weights)
```bash
python scripts/week4_qc_orchestrator.py \
    --image data/mvtec/bottle/test/good/000.png \
    --question "Is this product acceptable?" \
    --weights models/yolov10n-bottle/weights/best.pt
```

### Programmatic Query (Full Integration)
```python
from week4_qc_orchestrator import SentinelWeek4Orchestrator
import json

# Create orchestrator
orch = SentinelWeek4Orchestrator(
    yolo_weights=None,
    anthropic_model="claude-sonnet-4-20250514"  # Claude 3.5 Sonnet
)

# Process image
result = orch.process_query(
    image_path='data/mvtec/bottle/test/broken_large/000.png',
    question='What type of defect is visible? What action should we take?'
)

# Results include:
print(json.dumps({
    'defect_type': result.reasoning.get('defect_type'),
    'severity': result.reasoning.get('severity'),
    'confidence': result.reasoning.get('confidence'),
    'recommended_action': result.reasoning.get('recommended_action'),
    'retrieved_sops': len(result.retrieval)
}, indent=2))
```

---

## Output Format

Each query returns a `QCQueryResult` with:

```json
{
  "image_path": "data/mvtec/bottle/test/good/000.png",
  "question": "Assess quality...",
  "vision_summary": "Object detections and anomaly scores",
  "retrieval": [
    {
      "doc_id": "SOP-BOTTLE-001",
      "title": "Large Fracture Detection and Handling",
      "weighted_score": 0.856,
      "sources": [
        {"type": "text", "score": 0.85},
        {"type": "image", "score": 0.86, "defect_type": "broken_large"}
      ]
    }
  ],
  "reasoning": {
    "defect_type": "broken_large",
    "severity": "high",
    "root_cause": "Large fracture detected on bottle surface",
    "recommended_action": "Reject and quarantine for rework",
    "confidence": 0.92,
    "citations": ["SOP-BOTTLE-001"],
    "counterfactual": "If the crack were absent, the bottle would appear..."
  },
  "metadata": {
    "vision_details": {"yolo": {...}, "anomaly": {...}},
    "drift_warning": null,
    "max_image_similarity": 0.76
  }
}
```

---

## Features Breakdown

### 🎯 What Works Now (No API Key Needed)
- ✓ Image + text similarity search (Qdrant)
- ✓ SOP document retrieval
- ✓ Vision analysis (YOLO + PatchCore ready)
- ✓ Metadata tracking
- ✓ MLflow logging

### 🤖 What Activates With API Key
- ✓ Claude LLM reasoning
- ✓ Intelligent defect classification
- ✓ Root cause analysis
- ✓ Multimodal fusion reasoning

### 🔊 What's Always On
- ✓ Voice alerts (critical defects)
- ✓ Genealogy tracking
- ✓ Calibration logging
- ✓ Drift detection

---

## Test Data

**MVTec Bottle Dataset** available in: `data/mvtec/bottle/`

### Good Samples
- Folder: `test/good/` (20 images)
- Good bottles suitable for QA testing

### Defect Samples
- `test/broken_large/` - Large cracks (83 images)
- `test/broken_small/` - Minor damage (70 images)
- `test/contamination/` - Foreign material (73 images)

**Example Test Loop:**
```python
from pathlib import Path

test_dir = Path('data/mvtec/bottle/test')
for defect_type in ['good', 'broken_large', 'broken_small', 'contamination']:
    images = list((test_dir / defect_type).glob('*.png'))[:3]
    for img in images:
        result = orch.process_query(str(img))
        print(f'{defect_type:15} -> {result.reasoning["defect_type"]:12} (confidence: {result.reasoning["confidence"]:.2f})')
```

---

## Advanced Options

### Custom LLM Model
```python
orch = SentinelWeek4Orchestrator(
    anthropic_model="claude-opus-4-20250514"  # Claude 3 Opus
)
```

### Enable YOLO Vision Analysis
```python
orch = SentinelWeek4Orchestrator(
    yolo_weights="models/yolov10n-bottle/weights/best.pt"
)
```

### Custom Temperature / Generation
Modify in `_call_llm()`:
```python
# In week4_qc_orchestrator.py, line ~418
response = self.text_client.completions.create(
    model=self.anthropic_model,
    prompt=prompt,
    max_tokens_to_sample=2048,  # Increase for longer responses
    temperature=0.5,  # Lower = more deterministic, Higher = more creative
)
```

---

## Troubleshooting

### Issue: "No module named 'anthropic'"
```bash
pip install anthropic==0.89.0
```

### Issue: "ANTHROPIC_API_KEY is not set"
Feature is correctly disabled. To enable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python script.py  # Restart after setting env var
```

### Issue: Voice not working on Linux/Mac
SAPI5 is Windows-only. For cross-platform:
```python
# Use espeak (Linux) or say (Mac) instead
# Modify pyttsx3 engine selection in orchestrator
```

### Issue: Out of Memory with CLIP/SBERT
Models load on CPU by default (safe for most systems). If needed:
```python
# Modify in clip_embedder.py
self.device = 'cuda'  # If GPU available
```

---

## Performance Notes

| Component | Time | Memory |
|-----------|------|--------|
| Init (cold) | 5-10s | 2-3 GB |
| Init (warm) | 0.5s | 2-3 GB |
| Query (no LLM) | 1-2s | - |
| LLM Call | 2-5s | +100MB |
| Voice Alert | <1s | - |

---

## Monitoring & Logs

**Recent queries:** `data/week4_memory/week4_query_log.json`

**Genealogy graph:** `data/week4_memory/defect_graph.json`

**Calibration state:** `data/week4_memory/confidence_calibration.json`

**MLflow dashboard:**
```bash
mlflow ui
# Then visit: http://localhost:5000
```

---

## Next Steps

1. Set `ANTHROPIC_API_KEY` for full LLM features
2. Train YOLO on your defect dataset
3. Deploy orchestrator to production pipeline
4. Monitor via MLflow dashboard
5. Iterate on SOP documents based on feedback

---

**For detailed integration status, see:** `WEEK4_INTEGRATION_STATUS.md`
