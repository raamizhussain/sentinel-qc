#!/usr/bin/env python3
"""SENTINEL Week 4 - Multimodal QC Orchestration

This module builds a LangChain-capable QC reasoning pipeline that fuses:
- YOLOv10 vision detection
- PatchCore anomaly scoring
- CLIP image embeddings
- SBERT text embeddings
- Qdrant similarity retrieval
- LLM reasoning with Claude / Anthropic
- Federated memory, defect genealogy, drift prediction, calibration, and voice alerts
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from clip_embedder import CLIPEmbedder
except ImportError:
    CLIPEmbedder = None

try:
    from qdrant_db import QdrantVectorDB
except ImportError:
    QdrantVectorDB = None

try:
    from mlflow_utils import MLflowTracker
except ImportError:
    MLflowTracker = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from patchcore_detector import PatchCoreDetector
except ImportError:
    PatchCoreDetector = None


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_DIR = DATA_DIR / "week4_memory"
SOP_FILE = DATA_DIR / "synthetic_sops.json"
DEFECT_GRAPH_FILE = MEMORY_DIR / "defect_graph.json"
DEFECT_MEMORY_FILE = MEMORY_DIR / "defect_memory.json"
CALIBRATION_FILE = MEMORY_DIR / "confidence_calibration.json"
QUERY_LOG_FILE = MEMORY_DIR / "week4_query_log.json"

os.makedirs(MEMORY_DIR, exist_ok=True)


def safe_json_dump(value: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def safe_json_load(path: Path, default: Any):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


@dataclass
class QCQueryResult:
    image_path: str
    question: str
    vision_summary: str
    retrieval: List[Dict[str, Any]]
    reasoning: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SentinelWeek4Orchestrator:
    def __init__(self,
                 yolo_weights: Optional[str] = None,
                 text_model_name: str = "all-MiniLM-L6-v2",
                 memory_root: Path = MEMORY_DIR):
        self.memory_root = Path(memory_root)
        self.memory_root.mkdir(parents=True, exist_ok=True)

        self.text_model_name = text_model_name
        self.yolo_weights = yolo_weights

        self.clip = None
        self.text_embedder = None
        self.text_client = None
        self.voice_engine = None
        self.genealogy = None
        self.calibration_state = None
        self.drift_events: List[Dict[str, Any]] = []
        self.mlflow = None

        self.text_index = None
        self.image_index = None
        self.sop_documents: List[Dict[str, Any]] = []
        self.sop_lookup: Dict[str, Dict[str, Any]] = {}
        self.defect_to_sop: Dict[str, str] = {}

        self._initialize_components()

    def _initialize_components(self):
        print("[Week 4] Initializing SENTINEL components...")
        self._load_sop_documents()
        self._initialize_clip()
        self._initialize_text_encoder()
        self._initialize_qdrant_indices()
        self._load_genealogy()
        self._load_calibration()
        self._initialize_voice_alert()
        self._initialize_llm()
        self._initialize_mlflow()
        self._prepare_mappings()
        print("[Week 4] Initialization complete. Ready for queries.")

    def _load_sop_documents(self):
        self.sop_documents = safe_json_load(SOP_FILE, [])
        for sop in self.sop_documents:
            self.sop_lookup[sop["id"]] = sop
            self.defect_to_sop[sop["defect_type"]] = sop["id"]
        print(f"  - Loaded {len(self.sop_documents)} SOP documents")

    def _initialize_clip(self):
        if CLIPEmbedder is None:
            raise RuntimeError("CLIPEmbedder is unavailable. Install open_clip and clip_embedder.py must be present.")
        self.clip = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai", device="cpu")
        print("  - CLIP embedding model initialized")

    def _initialize_text_encoder(self):
        if SentenceTransformer is None:
            print("  - WARNING: sentence-transformers not installed; text retrieval will be disabled")
            self.text_embedder = None
            return
        self.text_embedder = SentenceTransformer(self.text_model_name)
        print(f"  - SBERT text encoder initialized: {self.text_model_name}")

    def _initialize_qdrant_indices(self):
        if QdrantVectorDB is None:
            raise RuntimeError("QdrantVectorDB is unavailable. Install qdrant-client and ensure qdrant_db.py is present.")

        self.text_index = QdrantVectorDB(
            collection_name="sentinel_sops",
            vector_size=384,
            storage_path=self.memory_root / "qdrant_text",
            in_memory=False
        )
        self.image_index = QdrantVectorDB(
            collection_name="sentinel_image_memory",
            vector_size=self.clip.embed_dim if self.clip else 512,
            storage_path=self.memory_root / "qdrant_image",
            in_memory=False
        )

        self._ensure_sop_index()
        self._ensure_image_memory()

    def _initialize_llm(self):
        if pipeline is None:
            print("  - WARNING: transformers not installed. LLM reasoning is disabled.")
            self.text_client = None
            return

        try:
            # Use a free local model for text generation
            self.text_client = pipeline("text2text-generation", model="google/flan-t5-small", device="cpu")
            print("  - LLM reasoning client initialized: google/flan-t5-small (free local model)")
        except Exception as e:
            print(f"  - WARNING: Failed to initialize local LLM: {e}")
            self.text_client = None

    def _initialize_mlflow(self):
        if MLflowTracker is None:
            print("  - WARNING: mlflow_utils not found. MLflow logging disabled.")
            self.mlflow = None
            return

        try:
            self.mlflow = MLflowTracker(experiment_name="sentinel_week4")
            print("  - MLflow tracker initialized")
        except Exception as e:
            print(f"  - WARNING: MLflow initialization failed: {e}")
            self.mlflow = None

    def _initialize_voice_alert(self):
        if pyttsx3 is None:
            print("  - WARNING: pyttsx3 not installed. Voice alert disabled.")
            self.voice_engine = None
            return

        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty("rate", 180)
            self.voice_engine.setProperty("volume", 0.9)
            print("  - Voice alert engine initialized")
        except Exception as e:
            print(f"  - WARNING: pyttsx3 initialization failed: {e}")
            self.voice_engine = None

    def _prepare_mappings(self):
        self.defect_to_sop = {sop["defect_type"]: sop["id"] for sop in self.sop_documents}

    def _load_genealogy(self):
        self.genealogy = safe_json_load(DEFECT_GRAPH_FILE, {"nodes": [], "edges": []})
        print(f"  - Loaded defect genealogy: {len(self.genealogy['nodes'])} nodes")

    def _load_calibration(self):
        self.calibration_state = safe_json_load(CALIBRATION_FILE, {"predictions": [], "labels": [], "ece": None})
        print("  - Loaded confidence calibration state")

    def _ensure_sop_index(self):
        if self.text_index.get_collection_stats()["vector_count"] == 0:
            if self.text_embedder is None:
                print("  - WARNING: Text index is empty and no SBERT model is available")
                return

            sop_texts = []
            sop_meta = []
            for sop in self.sop_documents:
                text = f"{sop['title']}. {sop['description']}. {sop['content']}"
                sop_texts.append(text)
                sop_meta.append({
                    "doc_id": sop["id"],
                    "title": sop["title"],
                    "defect_type": sop["defect_type"],
                    "product": sop["product"]
                })

            embeddings = self.text_embedder.encode(sop_texts, show_progress_bar=False, convert_to_numpy=True)
            self.text_index.add_images_batch(embeddings, [sop["id"] for sop in self.sop_documents], sop_meta)
            print(f"  - Indexed {len(self.sop_documents)} SOP documents into Qdrant")
        else:
            print("  - SOP text index already populated")

    def _ensure_image_memory(self):
        if self.image_index.get_collection_stats()["vector_count"] == 0:
            bottle_root = DATA_DIR / "mvtec" / "bottle"
            image_paths = []
            metadata = []
            for defect_category in ["good", "broken_large", "broken_small", "contamination"]:
                category_path = bottle_root / "test" / defect_category
                if category_path.exists():
                    files = sorted(category_path.glob("*.png"))[:5]
                    for image_path in files:
                        image_paths.append(str(image_path))
                        metadata.append({
                            "category": "bottle",
                            "defect_type": defect_category,
                            "source": "seed"
                        })

            if image_paths:
                embeddings = self.clip.embed_images(image_paths)
                self.image_index.add_images_batch(embeddings, image_paths, metadata)
                print(f"  - Seeded image memory with {len(image_paths)} bottle examples")
            else:
                print("  - No seed images found for image memory initialization")
        else:
            print("  - Image memory index already populated")

    def _build_feedback_context(self, query_image: str, question: Optional[str] = None) -> str:
        return f"Query image: {query_image}\nQuestion: {question or 'Describe defect and recommend action.'}\n"

    def _get_vision_summary(self, image_path: str) -> Tuple[str, Dict[str, Any]]:
        items = []
        details = {"yolo": None, "anomaly": None}

        if YOLO is not None and self.yolo_weights and Path(self.yolo_weights).exists():
            try:
                model = YOLO(self.yolo_weights)
                predictions = model.predict(source=image_path, imgsz=640, conf=0.2)
                boxes = []
                for p in predictions:
                    for box in p.boxes:
                        boxes.append({
                            "xyxy": box.xyxy.tolist(),
                            "confidence": float(box.conf[0]) if box.conf is not None else None,
                            "class_id": int(box.cls[0]) if box.cls is not None else None
                        })
                details["yolo"] = {"detections": boxes, "count": len(boxes)}
                items.append(f"Detected {len(boxes)} object(s) with YOLO")
            except Exception as e:
                items.append(f"YOLO inference failed: {e}")

        patchcore_model_path = Path(__file__).resolve().parent.parent / "models" / "patchcore_bottle.pkl"
        if PatchCoreDetector is not None and patchcore_model_path.exists():
            try:
                detector = PatchCoreDetector(backbone="resnet18", device="cpu")
                detector.load(str(patchcore_model_path))
                image = self._load_image_array(image_path)
                score, _ = detector.detect(image)
                details["anomaly"] = {"score": score}
                items.append(f"Anomaly score: {score:.3f}")
            except Exception as e:
                items.append(f"PatchCore detection unavailable: {e}")

        return ("; ".join(items) if items else "No vision analysis available."), details

    def _load_image_array(self, image_path: str) -> np.ndarray:
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _run_text_retrieval(self, question: str) -> List[Dict[str, Any]]:
        if self.text_embedder is None:
            return []
        query_embedding = self.text_embedder.encode(question, convert_to_numpy=True)
        return self.text_index.search(query_embedding, top_k=10)

    def _run_image_retrieval(self, image_path: str) -> List[Dict[str, Any]]:
        image_embedding = self.clip.embed_image(image_path)
        return self.image_index.search(image_embedding, top_k=10)

    def _fuse_retrievals(self, text_hits: List[Dict[str, Any]], image_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}

        for hit in text_hits:
            doc_id = hit["image_path"]
            fused.setdefault(doc_id, {
                "doc_id": doc_id,
                "title": hit.get("defect_type", doc_id),
                "weighted_score": 0.0,
                "sources": []
            })
            fused[doc_id]["weighted_score"] += 0.4 * hit.get("similarity", 0.0)
            fused[doc_id]["sources"].append({"type": "text", "score": hit.get("similarity", 0.0)})

        for hit in image_hits:
            defect_type = hit.get("defect_type", "unknown")
            sop_id = self.defect_to_sop.get(defect_type)
            if sop_id is None:
                continue
            fused.setdefault(sop_id, {
                "doc_id": sop_id,
                "title": self.sop_lookup.get(sop_id, {}).get("title", sop_id),
                "weighted_score": 0.0,
                "sources": []
            })
            fused[sop_id]["weighted_score"] += 0.6 * hit.get("similarity", 0.0)
            fused[sop_id]["sources"].append({"type": "image", "score": hit.get("similarity", 0.0), "defect_type": defect_type})

        results = sorted(fused.values(), key=lambda x: x["weighted_score"], reverse=True)
        return results[:5]

    def _generate_reasoning_prompt(self, question: str, vision_summary: str, top_contexts: List[Dict[str, Any]]) -> str:
        context_text = "\n".join([
            f"{idx+1}. {self.sop_lookup.get(doc['doc_id'], {}).get('title', doc['doc_id'])} (score={doc['weighted_score']:.3f})"
            for idx, doc in enumerate(top_contexts)
        ])
        sop_details = "\n\n".join([
            f"SOP: {self.sop_lookup.get(doc['doc_id'], {}).get('title', doc['doc_id'])}\n{self.sop_lookup.get(doc['doc_id'], {}).get('description', '')[:200]}"
            for doc in top_contexts
        ])

        prompt = f"You are a factory quality control expert. Analyze the defect using the image analysis results and the retrieved SOP documents below. Answer in valid JSON only with the following fields:\n" \
                 "defect_type, severity, root_cause, recommended_action, confidence, citations, counterfactual.\n" \
                 "Provide concise reasoning, cite the SOP titles, and include a counterfactual description of what the product would look like if the defect were absent.\n\n" \
                 f"IMAGE SUMMARY: {vision_summary}\n\n" \
                 f"RETRIEVED SOPS:\n{context_text}\n\n" \
                 f"SOP DETAIL SUMMARY:\n{sop_details}\n\n" \
                 f"QUESTION: {question or 'Recommend QC action based on the image and SOP context.'}\n"
        return prompt

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if self.text_client is None:
            return {
                "defect_type": "unknown",
                "severity": "unknown",
                "root_cause": "LLM disabled",
                "recommended_action": "Manual review required",
                "confidence": 0.0,
                "citations": [],
                "counterfactual": "No LLM available to generate counterfactual reasoning."
            }

        try:
            # Format prompt for Flan-T5
            formatted_prompt = f"Analyze the defect and provide a JSON response with fields: defect_type, severity, root_cause, recommended_action, confidence, citations, counterfactual. {prompt}"
            response = self.text_client(formatted_prompt, max_length=512, do_sample=False)
            output = response[0]['generated_text'] if response else ""
            if not output:
                raise ValueError("Local LLM did not generate output")
            return self._parse_llm_json(output)
        except Exception as e:
            print(f"  - WARNING: LLM call failed: {e}")
            return {
                "defect_type": "unknown",
                "severity": "unknown",
                "root_cause": f"LLM call failed: {e}",
                "recommended_action": "Manual review required",
                "confidence": 0.0,
                "citations": [],
                "counterfactual": "No counterfactual generated."
            }

    @staticmethod
    def _parse_llm_json(text: str) -> Dict[str, Any]:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        # Fallback: try to extract JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        return {
            "defect_type": "unknown",
            "severity": "unknown",
            "root_cause": "Could not parse LLM output.",
            "recommended_action": text.strip(),
            "confidence": 0.0,
            "citations": [],
            "counterfactual": "Unable to extract counterfactual."
        }

    def _run_voice_alert(self, severity: str, defect_type: str, product_id: str):
        if self.voice_engine is None or severity.lower() != "critical":
            return

        def speak():
            message = f"Critical defect detected: {defect_type} on product {product_id}."
            try:
                self.voice_engine.say(message)
                self.voice_engine.runAndWait()
            except Exception as e:
                print(f"  - Voice alert failed: {e}")

        thread = threading.Thread(target=speak, daemon=True)
        thread.start()

    def _log_calibration(self, confidence: float, label: Optional[int] = None):
        if label is not None:
            self.calibration_state.setdefault("predictions", []).append(confidence)
            self.calibration_state.setdefault("labels", []).append(label)
            self.calibration_state["ece"] = self.compute_ece(
                np.array(self.calibration_state["predictions"]),
                np.array(self.calibration_state["labels"])
            )
            safe_json_dump(self.calibration_state, CALIBRATION_FILE)

    @staticmethod
    def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        if len(confidences) == 0 or len(label := labels) == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(mask) == 0:
                continue
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(labels[mask])
            ece += np.abs(bin_conf - bin_acc) * np.sum(mask) / len(confidences)
        return float(ece)

    def _update_drift(self, defect_type: str):
        timestamp = datetime.utcnow().isoformat()
        self.drift_events.append({"timestamp": timestamp, "defect_type": defect_type})
        self._save_drift_history()

        if LinearRegression is None:
            return None

        last_24h = [event for event in self.drift_events if datetime.fromisoformat(event["timestamp"]) > datetime.utcnow() - timedelta(hours=24)]
        if len(last_24h) < 3:
            return None

        counts = {}
        for event in last_24h:
            hour = datetime.fromisoformat(event["timestamp"]).replace(minute=0, second=0, microsecond=0).isoformat()
            counts[hour] = counts.get(hour, 0) + 1

        hours = sorted(counts.keys())
        X = np.arange(len(hours)).reshape(-1, 1)
        y = np.array([counts[h] for h in hours], dtype=float)
        model = LinearRegression().fit(X, y)
        slope = float(model.coef_[0])
        if slope > 0.1:
            warning = f"Predicted drift warning: defect arrival rate increasing (slope={slope:.3f})."
            print(f"  - {warning}")
            return warning
        return None

    def _save_drift_history(self):
        path = self.memory_root / "drift_events.json"
        safe_json_dump(self.drift_events, path)

    def _save_genealogy(self):
        safe_json_dump(self.genealogy, DEFECT_GRAPH_FILE)

    def visualize_genealogy(self, output_path: Optional[Path] = None):
        if nx is None:
            print("  - NetworkX is not installed; genealogy visualization disabled.")
            return
        if output_path is None:
            output_path = self.memory_root / "defect_genealogy.png"

        try:
            import matplotlib.pyplot as plt

            graph = nx.DiGraph()
            for node in self.genealogy.get("nodes", []):
                graph.add_node(node["id"], label=node.get("defect_type", "unknown"))
            for edge in self.genealogy.get("edges", []):
                graph.add_edge(edge["from"], edge["to"], weight=edge.get("weight", 0.0))

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph, seed=42)
            labels = {node: data.get("label", node) for node, data in graph.nodes(data=True)}
            nx.draw(graph, pos, with_labels=True, node_size=800, node_color="#4C72B0", font_size=8)
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
            plt.title("SENTINEL Defect Genealogy Graph")
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  - Defect genealogy visualization saved to {output_path}")
        except Exception as e:
            print(f"  - Genealogy visualization failed: {e}")

    def _save_query_log(self, result: Dict[str, Any]):
        logs = safe_json_load(QUERY_LOG_FILE, [])
        logs.append(result)
        safe_json_dump(logs, QUERY_LOG_FILE)

    def _save_defect_memory(self, metadata: Dict[str, Any]):
        memory = safe_json_load(DEFECT_MEMORY_FILE, [])
        memory.append(metadata)
        safe_json_dump(memory, DEFECT_MEMORY_FILE)

    def _add_to_genealogy(self, event_id: str, similarity: float, event_metadata: Dict[str, Any]):
        if nx is None:
            return
        self.genealogy.setdefault("nodes", []).append({"id": event_id, **event_metadata})
        for node in self.genealogy.get("nodes", []):
            if node["id"] == event_id:
                continue
            if node.get("defect_type") == event_metadata.get("defect_type") and similarity > 0.8:
                self.genealogy.setdefault("edges", []).append({"from": node["id"], "to": event_id, "weight": similarity})
        self._save_genealogy()

    def _log_mlflow(self, query_result: QCQueryResult):
        if self.mlflow is None:
            return
        run_name = f"week4_query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        try:
            self.mlflow.start_run(run_name=run_name, tags={"phase": "week4", "workflow": "qc_orchestration"})
            self.mlflow.log_params({
                "image_path": query_result.image_path,
                "question": query_result.question,
                "vision_summary": query_result.vision_summary[:120]
            })
            self.mlflow.log_metrics({
                "retrieval_count": len(query_result.retrieval),
                "confidence": float(query_result.reasoning.get("confidence", 0.0))
            })
            artifact_path = self.memory_root / f"mlflow_query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            safe_json_dump(query_result.__dict__, artifact_path)
            self.mlflow.log_artifact(str(artifact_path), artifact_path="week4_queries")
        except Exception as e:
            print(f"  - MLflow logging failed: {e}")
        finally:
            self.mlflow.end_run()

    def _generate_description_for_new_defect(self, image_path: str, similarity: float) -> str:
        defect_type = Path(image_path).parent.name
        return (
            f"New defect event from image {Path(image_path).name} with low similarity ({similarity:.3f}) to known cases. "
            f"Detected defect category: {defect_type}. "
            "This event is being added to the federated knowledge base for future matching."
        )

    def process_query(self, image_path: str, question: Optional[str] = None) -> QCQueryResult:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if question is None:
            question = "Identify the product defect, severity, root cause, recommended action, and confidence."

        vision_summary, vision_details = self._get_vision_summary(image_path)
        text_results = self._run_text_retrieval(question)
        image_results = self._run_image_retrieval(image_path)
        fused_results = self._fuse_retrievals(text_results, image_results)
        prompt = self._generate_reasoning_prompt(question, vision_summary, fused_results)
        reasoning = self._call_llm(prompt)

        max_similarity = max([hit.get("similarity", 0.0) for hit in image_results] or [0.0])
        if max_similarity < 0.6:
            new_point_id = self.image_index.add_image(
                self.clip.embed_image(image_path),
                image_path,
                metadata={
                    "category": "unknown",
                    "defect_type": Path(image_path).parent.name,
                    "is_good": False,
                    "source": "federated_memory",
                    "description": self._generate_description_for_new_defect(image_path, max_similarity)
                }
            )
            event_id = str(new_point_id)
            self._add_to_genealogy(event_id, max_similarity, {
                "defect_type": Path(image_path).parent.name,
                "image_path": str(image_path),
                "similarity": max_similarity,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._save_defect_memory({
                "event_id": event_id,
                "image_path": str(image_path),
                "similarity": max_similarity,
                "description": self._generate_description_for_new_defect(image_path, max_similarity),
                "created_at": datetime.utcnow().isoformat()
            })

        self._run_voice_alert(str(reasoning.get("severity", "unknown")), reasoning.get("defect_type", "unknown"), Path(image_path).stem)
        drift_warning = self._update_drift(reasoning.get("defect_type", "unknown"))
        self._log_calibration(float(reasoning.get("confidence", 0.0)))

        query_result = QCQueryResult(
            image_path=str(image_path),
            question=question,
            vision_summary=vision_summary,
            retrieval=fused_results,
            reasoning=reasoning,
            metadata={
                "vision_details": vision_details,
                "drift_warning": drift_warning,
                "max_image_similarity": max_similarity
            }
        )

        self._save_query_log(query_result.__dict__)
        self._log_mlflow(query_result)
        return query_result


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the SENTINEL Week 4 QC orchestrator")
    parser.add_argument("--image", type=str, help="Path to a defect image")
    parser.add_argument("--question", type=str, default=None, help="Optional QC question")
    parser.add_argument("--weights", type=str, default=str(Path(__file__).resolve().parent.parent / "models" / "yolov10n-bottle" / "weights" / "best.pt"), help="Path to YOLO weights")
    args = parser.parse_args()

    orchestrator = SentinelWeek4Orchestrator(yolo_weights=args.weights)
    result = orchestrator.process_query(args.image, args.question)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
