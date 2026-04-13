#!/usr/bin/env python3
"""
SENTINEL Week 3 - Text RAG Pipeline with Synthetic SOPs
Generates synthetic maintenance SOPs and builds text-image retrieval system
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import json
from datetime import datetime
import numpy as np

print("\n" + "="*80)
print("SENTINEL WEEK 3 - TEXT RAG PIPELINE SETUP")
print("="*80 + "\n")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_keywords(text):
    """Extract important keywords from SOP content"""
    keywords = set()
    # Simple keyword extraction (in production, use TFIDF or NLP)
    important_terms = [
        'SCOPE', 'IDENTIFICATION', 'INSPECTION', 'ACCEPTANCE',
        'REJECT', 'ACCEPT', 'RISK', 'PROCEDURE', 'DOCUMENTATION'
    ]
    for term in important_terms:
        if term in text:
            keywords.add(term.lower())
    return list(keywords)

def chunk_document(text, chunk_size=200):
    """Split document into chunks of words"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ============================================================================
# STEP 1: Define Synthetic SOP Templates
# ============================================================================

print("[STEP 1] Defining Synthetic SOP Templates...\n")

sop_templates = {
    "bottle": {
        "product": "Glass Bottle",
        "defects": {
            "broken_large": {
                "title": "SOP-BOTTLE-001: Large Fracture Detection and Handling",
                "description": "Procedure for identifying and handling bottles with large structural fractures",
                "content": """
SCOPE: This SOP applies to detection and management of large fractures (>5mm) in glass bottles.

IDENTIFICATION:
1. Visual inspection under bright light (>500 lux)
2. Listen for acoustic signatures (slight tapping)
3. Temperature differential detection (thermography optional)

LOCATION PATTERNS:
- Neck fractures: Primary weak point, leads to rapid propagation
- Body fractures: Typically vertical, originating from stress concentration
- Base fractures: Usually circular, from thermal shock or impact

RISK ASSESSMENT:
- Contamination risk: HIGH (glass particles)
- Leakage risk: HIGH (structural compromise)
- Consumer safety: CRITICAL

REMEDIATION:
1. Isolate defective batch immediately
2. Secondary inspection of adjacent units (n+/-2)
3. Root cause analysis (thermal, mechanical, material fatigue)
4. Quarantine for recycling (do not repair)

DOCUMENTATION:
- Photo evidence (top, side, bottom views)
- Lot number and production timestamp
- Inspector signature and date
- QC disposition code: REJECT
                """.strip()
            },
            "broken_small": {
                "title": "SOP-BOTTLE-002: Minor Defect Triage and Acceptance Criteria",
                "description": "Procedure for small cracks and chips in glass bottles",
                "content": """
SCOPE: This SOP applies to detection of minor defects (<3mm) that may be acceptable.

IDENTIFICATION:
1. Magnified visual inspection (10x magnification)
2. Surface palpation (gloved touch inspection)
3. Dimensional verification with precision calipers

DEFECT CLASSIFICATION:
- Micro-cracks: <1mm length, surface-level only - ACCEPT
- Small chips: <3mm diameter, not at stress points - CONDITIONAL ACCEPT
- Surface scratches: Cosmetic only - ACCEPT

STRESS POINT DEFINITION:
- Bottle neck junction
- Base impact zone
- Label application area (stress concentration)

CONDITIONAL ACCEPTANCE CRITERIA:
- Defect must be >5mm from bottle rim
- Defect must be <15mm from base
- No sharp edges (must be smooth for handling safety)

CUSTOMER COMMUNICATION:
- Inform customers of minor defects if >2% of batch affected
- Provide photographic documentation
- Offer batch replacement guarantee if needed

DOCUMENTATION:
- QC disposition code: ACCEPT or CONDITIONAL-ACCEPT
                """.strip()
            },
            "contamination": {
                "title": "SOP-BOTTLE-003: Foreign Material Detection Protocol",
                "description": "Procedure for identifying and handling contaminated bottles",
                "content": """
SCOPE: Detection and handling of foreign material inside or on bottle surface.

CONTAMINATION TYPES:
1. Internal contamination: particles, liquid residue, organic matter
2. External contamination: dust, labels improperly applied, chemical residue
3. Structural contamination: embedded glass or metal fragments

DETECTION METHODS:
- Visual inspection (internal light source - 1000 lux minimum)
- Shake test (vigorous shaking to mobilize internal particles)
- Weight differential analysis (full vs empty) - threshold ±2%
- Spectral analysis (optional, for chemical contamination)

TOLERANCE LEVELS:
- Transparent bottles: ANY visible internal particle = REJECT
- Opaque bottles: <5mm particles acceptable if not at contact surfaces
- Surface contamination: >10% coverage = REJECT

REMEDIATION:
1. Internal contamination: Rinse, re-inspect, or discard
2. External contamination: Clean with soft cloth and approved solvent
3. Persistent contamination: Quarantine for investigation

TRACEABILITY:
- Source identification (track back to production line)
- Batch hold and investigation
- Preventive maintenance scheduling for equipment

DOCUMENTATION:
- Photo of contamination (magnified if necessary)
- Type of foreign material identified
- Remediation action taken
- QC disposition: REJECT or REWORK
                """.strip()
            }
        }
    },
    "cable": {
        "product": "Electronic Cable Assembly",
        "defects": {
            "cut_outer_insulation": {
                "title": "SOP-CABLE-001: Outer Insulation Integrity Assessment",
                "description": "Procedure for evaluating cuts and abrasions in outer jacket",
                "content": """
SCOPE: Detection and evaluation of outer insulation damage in cable assemblies.

INSPECTION EQUIPMENT:
- Visual inspection (bright field illumination)
- Continuity tester (optional, if inner conductors exposed)
- Insulation resistance meter (>20 megohms @ 500V DC)
- Micrometer for depth measurement

DEFECT CLASSIFICATION:
- Surface abrasion: Only cosmetic, no conductor exposure - ACCEPT
- Shallow cut: <50% insulation depth - CONDITIONAL ACCEPT
- Full cut: Conductor exposed, >50% depth - REJECT

RISK ASSESSMENT:
- Electrical safety: Short circuit risk if wet environment
- Signal integrity: Crosstalk if shielding compromised
- Physical safety: Shock hazard if high voltage application

ACCEPTANCE CRITERIA:
- No exposure of inner conductor
- No exposed shielding layer
- Insulation resistance remains >10 megohms
- Length of cut <1 inch (25mm) for conditional acceptance

REPAIR OPTIONS:
- Electrical tape wrapping (temporary, <6 months)
- Heat-shrink tube (preferred method)
- Cable segment replacement (if damage >2 inches)

DOCUMENTATION:
- Cut location (distance from connector, cm)
- Depth measurement (mm)
- Photo evidence with scale reference
- Repair method and materials used
                """.strip()
            },
            "bent_wire": {
                "title": "SOP-CABLE-002: Wire Strain Relief and Bend Radius Compliance",
                "description": "Procedure for assessing wire bending damage and mechanical stress",
                "content": """
SCOPE: Evaluation of bent, kinked, or damaged wires in cable assemblies.

MECHANICAL INSPECTION:
1. Visual inspection of wire routing through strain relief
2. Bend radius measurement (minimum radius per cable specification)
3. Continuity testing on individual conductors
4. Resistance measurement across conductor length

DEFECT TYPES:
- Permanent kink: Conductor cross-section compromised, resistance increased
- Sharp bend: Radius <2x cable diameter - Risk of internal wire breakage
- Stress mark: Visible whitening or cracking in wire insulation

BEND RADIUS REQUIREMENTS:
- Stranded copper: Minimum 4x cable outer diameter
- Twisted pair: Minimum 3x cable outer diameter
- Ribbon cable: Minimum 2x cable width

ELECTRICAL TESTING:
- DC resistance: Measure vs. baseline (tolerance ±5%)
- High-frequency insertion loss: <1dB deviation acceptable
- Impedance: Maintain 50-100 ohm specification

ACCEPTANCE CRITERIA:
- Bend radius compliant with specification
- No white stress marks or surface cracking
- Electrical parameters within tolerance (resistance, impedance)
- Individual conductor continuity verified

REMEDIATION:
- Straighten wire carefully if kink is recent
- Replace conductor if resistance increased >10%
- Re-route cable to comply with bend radius

DOCUMENTATION:
- Photo of bend (angle reference, measurement tool shown)
- Resistance measurements (ohms)
- Bend radius calculation
- Acceptance or rejection decision
                """.strip()
            }
        }
    }
}

print(f"  [OK] Loaded {len(sop_templates)} product categories")
for product, data in sop_templates.items():
    defect_count = len(data.get('defects', {}))
    print(f"      - {data['product']}: {defect_count} SOP templates")

# ============================================================================
# STEP 2: Generate Synthetic SOP Documents
# ============================================================================

print("\n[STEP 2] Generating Synthetic SOP Documents...\n")

sop_documents = []
sop_dir = Path("data/synthetic_sops")
sop_dir.mkdir(parents=True, exist_ok=True)

for product_id, product_data in sop_templates.items():
    for defect_type, sop_info in product_data.get('defects', {}).items():
        sop_doc = {
            "id": f"SOP-{product_id.upper()}-{defect_type}",
            "product": product_data['product'],
            "defect_type": defect_type,
            "title": sop_info['title'],
            "description": sop_info['description'],
            "content": sop_info['content'],
            "generated_at": datetime.now().isoformat(),
            "length_tokens": len(sop_info['content'].split()),
            "keywords": extract_keywords(sop_info['content'])
        }
        sop_documents.append(sop_doc)
        
        # Save individual SOP as text file
        sop_filename = sop_dir / f"{sop_doc['id']}.txt"
        with open(sop_filename, 'w') as f:
            f.write(f"Title: {sop_info['title']}\n")
            f.write(f"Product: {product_data['product']}\n")
            f.write(f"Defect Type: {defect_type}\n")
            f.write("-" * 70 + "\n\n")
            f.write(sop_info['content'])
        
        print(f"  - {sop_doc['id']}: \"{sop_info['title'][:60]}...\"")

print(f"\n  [OK] Generated {len(sop_documents)} SOP documents")
print(f"  [OK] Saved to: {sop_dir}")

# ============================================================================
# STEP 3: Generate Text Embeddings (Sentence Transformer preparation)
# ============================================================================

print("\n[STEP 3] Preparing Text Embedding System...\n")

text_embedding_config = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "model_size_mb": 80,
    "device": "cpu",
    "description": "Lightweight sentence transformer for fast text embeddings"
}

print(f"  Text Embedding Model: {text_embedding_config['model_name']}")
print(f"  Embedding dimension: {text_embedding_config['embedding_dim']}")
print(f"  Model size: {text_embedding_config['model_size_mb']} MB")
print(f"  [OK] Sentence Transformer ready for deployment")

# ============================================================================
# STEP 4: Build RAG Document Index Structure
# ============================================================================

print("\n[STEP 4] Building RAG Document Index...\n")

rag_index = {
    "version": "1.0",
    "created_at": datetime.now().isoformat(),
    "total_documents": len(sop_documents),
    "documents": [],
    "metadata": {
        "embedding_model": text_embedding_config['model_name'],
        "embedding_dimension": text_embedding_config['embedding_dim'],
        "vector_db": "qdrant",
        "retrieval_method": "semantic similarity"
    }
}

for i, sop in enumerate(sop_documents):
    # Prepare document chunks (split long SOPs into sections)
    chunks = chunk_document(sop['content'], chunk_size=200)
    
    doc_entry = {
        "doc_id": sop['id'],
        "title": sop['title'],
        "product": sop['product'],
        "defect_type": sop['defect_type'],
        "chunks": len(chunks),
        "total_tokens": sop['length_tokens'],
        "keywords": sop['keywords'],
        "chunks_sample": chunks[:2]  # Store first 2 chunks as sample
    }
    rag_index['documents'].append(doc_entry)
    
    print(f"  - {sop['id']}: {len(chunks)} chunks, {sop['length_tokens']} tokens")

# ============================================================================
# STEP 5: Create RAG Query Examples
# ============================================================================

print("\n[STEP 5] Defining RAG Query Examples...\n")

query_examples = [
    {
        "query": "What should I do if I find a large crack in a bottle?",
        "expected_doc": "SOP-BOTTLE-001",
        "query_type": "procedural",
        "context": "User finding defect in product"
    },
    {
        "query": "How do I assess if a cable's outer insulation damage is acceptable?",
        "expected_doc": "SOP-CABLE-001",
        "query_type": "acceptance_criteria",
        "context": "QC decision making"
    },
    {
        "query": "Detection and handling of foreign particles inside bottles",
        "expected_doc": "SOP-BOTTLE-003",
        "query_type": "contamination",
        "context": "Contamination response"
    },
    {
        "query": "Wire bend radius compliance for cable assemblies",
        "expected_doc": "SOP-CABLE-002",
        "query_type": "mechanical_spec",
        "context": "Quality specification"
    },
]

print(f"  [OK] Created {len(query_examples)} query examples for RAG testing")
for ex in query_examples:
    print(f"      - Query: \"{ex['query'][:50]}...\" -> {ex['expected_doc']}")

# ============================================================================
# STEP 6: Integration with Multimodal Pipeline
# ============================================================================

print("\n[STEP 6] Planning Multimodal + Text RAG Integration...\n")

integration_plan = {
    "architecture": {
        "stage1_vision": "YOLOv10 detects defect + PatchCore scores anomaly",
        "stage2_retrieval": "CLIP + Text embedding finds relevant SOPs",
        "stage3_reasoning": "LLM reads SOP + image context, generates response",
        "stage4_feedback": "User confirms diagnosis, update training data"
    },
    "data_flow": [
        "Input: Factory image with potential defect",
        "Step 1: Vision models analyze image, extract features",
        "Step 2: Generate CLIP image embedding",
        "Step 3: Query RAG index: 'What defect does this match?'",
        "Step 4: Retrieve top-3 relevant SOP documents",
        "Step 5: LLM synthesizes: SOP context + image analysis -> recommendation",
        "Output: Structured QC decision with confidence score"
    ],
    "latency_breakdown": {
        "vision_inference": "0.5s (YOLO + PatchCore)",
        "embedding_generation": "0.2s (CLIP + sentence-transformer)",
        "rag_retrieval": "0.1s (vector similarity search)",
        "llm_inference": "2-5s (depends on context length)",
        "total_estimated": "3-6s per image"
    }
}

print("  Multimodal RAG Integration Plan:")
for i, step in enumerate(integration_plan['data_flow'], 1):
    print(f"    Step {i}: {step}")

print(f"\n  Estimated latency: {integration_plan['latency_breakdown']['total_estimated']}")

# ============================================================================
# STEP 7: Save Configuration and Metadata
# ============================================================================

print("\n[STEP 7] Saving Configuration and Metadata...\n")

# Save SOP documents to JSON
sop_file = Path("data/synthetic_sops.json")
with open(sop_file, 'w') as f:
    json.dump(sop_documents, f, indent=2)
print(f"  [OK] SOPs saved to {sop_file}")

# Save RAG index
rag_file = Path("data/rag_index.json")
with open(rag_file, 'w') as f:
    json.dump(rag_index, f, indent=2)
print(f"  [OK] RAG index saved to {rag_file}")

# Save query examples
queries_file = Path("data/rag_query_examples.json")
with open(queries_file, 'w') as f:
    json.dump(query_examples, f, indent=2)
print(f"  [OK] Query examples saved to {queries_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEK 3 SETUP COMPLETE")
print("="*80)

summary = f"""
Text RAG Pipeline - Ready for Implementation

SYNTHETIC DOCUMENTS GENERATED:
  - Total SOPs: {len(sop_documents)}
  - Product types: {len(sop_templates)}
  - Total tokens: {sum(d['length_tokens'] for d in sop_documents)}
  - Storage: {sop_dir}

RAG CONFIGURATION:
  - Embedding model: {text_embedding_config['model_name']}
  - Embedding dimensions: {text_embedding_config['embedding_dim']}
  - Vector database: Qdrant
  - Query examples: {len(query_examples)}

INTEGRATION POINTS:
  - Vision: YOLOv10 + PatchCore + CLIP (Week 2)
  - Text: Sentence-transformers + LLM
  - Search: Qdrant semantic similarity
  - Synthesis: LangChain orchestration (Week 4)

NEXT STEPS:
  1. Install sentence-transformers library
  2. Generate embeddings for all SOP documents
  3. Populate Qdrant with text embeddings
  4. Test RAG retrieval on query examples
  5. Integrate with vision pipeline (Week 4)
  6. Add LLM reasoning layer (Week 4)

READY FOR WEEK 4: Full Multimodal RAG Integration
"""

print(summary)
print("="*80 + "\n")

# Save summary
summary_file = Path("data/week3_setup_summary.txt")
with open(summary_file, 'w') as f:
    f.write(summary)

print(f"Summary saved to {summary_file}")
