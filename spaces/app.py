import gradio as gr
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

try:
    from week4_qc_orchestrator import Week4QCOrchestrator
    orchestrator = Week4QCOrchestrator()
    print("✓ SENTINEL orchestrator initialized")
except Exception as e:
    print(f"✗ Error initializing orchestrator: {e}")
    orchestrator = None

def process_defect_image(image):
    """
    Process a defect image through the SENTINEL pipeline
    Returns: defect_type, severity, root_cause, confidence, counterfactual
    """
    if orchestrator is None:
        return "Error", "N/A", "Orchestrator not initialized", "0%", "N/A"

    try:
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, format='PNG')
            temp_path = tmp_file.name

        # Process through orchestrator
        result = orchestrator.process_query({
            'image_path': temp_path,
            'query_type': 'defect_analysis'
        })

        # Clean up temp file
        os.unlink(temp_path)

        # Extract results
        defect_type = result.get('defect_type', 'Unknown')
        severity = result.get('severity', 'Unknown')
        root_cause = result.get('root_cause', 'Unknown')
        confidence = f"{result.get('confidence', 0)*100:.1f}%"
        counterfactual = result.get('counterfactual', 'N/A')

        return defect_type, severity, root_cause, confidence, counterfactual

    except Exception as e:
        return f"Error: {str(e)}", "N/A", "Processing failed", "0%", "N/A"

# Create Gradio interface
with gr.Blocks(title="SENTINEL - AI Quality Control System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 SENTINEL - AI Quality Control System

    **Upload a defect image to analyze:**
    - Detects defect type and severity
    - Identifies root cause
    - Provides confidence score
    - Suggests counterfactual analysis

    *Powered by multimodal AI: YOLOv10, CLIP, SBERT, and LLM reasoning*
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Defect Image", type="pil")
            analyze_btn = gr.Button("🔍 Analyze Defect", variant="primary")

        with gr.Column():
            defect_type = gr.Textbox(label="Defect Type", interactive=False)
            severity = gr.Textbox(label="Severity", interactive=False)
            root_cause = gr.Textbox(label="Root Cause", interactive=False)
            confidence = gr.Textbox(label="Confidence", interactive=False)
            counterfactual = gr.Textbox(label="Counterfactual", interactive=False)

    analyze_btn.click(
        fn=process_defect_image,
        inputs=image_input,
        outputs=[defect_type, severity, root_cause, confidence, counterfactual]
    )

    gr.Markdown("""
    ---
    **About SENTINEL:**
    - Multimodal defect analysis using vision + text + reasoning
    - Supports MVTec dataset categories (bottles, cables, etc.)
    - Real-time processing with vector search and genealogy tracking
    - Built for industrial quality control applications
    """)

if __name__ == "__main__":
    demo.launch()