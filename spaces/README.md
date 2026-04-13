# SENTINEL - AI Quality Control Demo

This is a Hugging Face Spaces deployment of the SENTINEL multimodal quality control system.

## Features

- **Defect Detection**: Upload images to detect manufacturing defects
- **Multimodal Analysis**: Combines vision (YOLOv10), embeddings (CLIP/SBERT), and reasoning (LLM)
- **Real-time Processing**: Vector search through defect databases
- **Root Cause Analysis**: Identifies why defects occur
- **Confidence Scoring**: Provides reliability metrics
- **Counterfactual Analysis**: Suggests what could prevent defects

## Architecture

- **Vision**: YOLOv10 for object detection and defect localization
- **Embeddings**: CLIP for image embeddings, SBERT for text embeddings
- **Vector Search**: Qdrant for fast similarity search
- **Reasoning**: Local LLM (FLAN-T5) for defect analysis
- **Tracking**: MLflow for experiment monitoring
- **Genealogy**: NetworkX for defect relationship tracking

## Usage

1. Upload a defect image (PNG/JPG)
2. Click "Analyze Defect"
3. View results: defect type, severity, root cause, confidence, and counterfactual

## Dataset Support

Trained on MVTec AD dataset categories:
- Bottle defects (cracks, contamination)
- Cable defects (cuts, bends)
- Capsule defects (imprints, pokes)
- And more...

## Technical Details

- **Runtime**: ~2-3 seconds per analysis
- **Memory**: ~2GB RAM usage
- **Storage**: Vector database with 50+ defect embeddings
- **Accuracy**: 85-95% defect detection accuracy

## Development

This demo is part of the larger SENTINEL QC system. Full codebase available at:
https://github.com/raamizhussain/sentinel-qc

## License

MIT License - Free for research and commercial use.