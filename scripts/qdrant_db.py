"""
Qdrant Vector Database Integration
Manages image/text embeddings in Qdrant for fast similarity search
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class QdrantVectorDB:
    """
    Wrapper around Qdrant for efficient vector similarity search
    Supports in-memory and persistent storage
    """
    
    def __init__(self, collection_name: str = "sentinel-defects", 
                 vector_size: int = 512,
                 storage_path: str = None,
                 in_memory: bool = False):
        """
        Initialize Qdrant client
        
        Args:
            collection_name: Collection name in Qdrant
            vector_size: Embedding dimension
            storage_path: Path for persistent storage (if None, in-memory)
            in_memory: Use in-memory storage
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        if in_memory:
            self.client = QdrantClient(":memory:")
            print("OK Qdrant in-memory database created")
        elif storage_path:
            storage_path = Path(storage_path) / self.collection_name
            storage_path.mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=str(storage_path))
                print(f"OK Qdrant database at {storage_path}")
            except RuntimeError as exc:
                error_message = str(exc).lower()
                if "already accessed" in error_message or "permission denied" in error_message:
                    print(f"  - WARNING: Qdrant storage at {storage_path} is locked or unavailable: {exc}")
                    print("  - Falling back to in-memory Qdrant database for this session")
                    self.client = QdrantClient(":memory:")
                    self.in_memory = True
                else:
                    raise
        else:
            self.client = QdrantClient(":memory:")
            print("OK Qdrant in-memory database")
        
        self.in_memory = in_memory or getattr(self, "in_memory", False)
        
        # Create collection if doesn't exist
        self._init_collection()
        
        # Track metadata
        self.metadata_store = {}
    
    def _init_collection(self):
        """Create collection with proper configuration"""
        try:
            self.client.get_collection(self.collection_name)
            print(f"OK Using existing collection: {self.collection_name}")
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"OK Created collection: {self.collection_name}")
    
    def add_image(self, embedding: np.ndarray, 
                  image_path: str, 
                  metadata: Dict = None,
                  point_id: Optional[int] = None) -> int:
        """
        Add single image embedding to database
        
        Args:
            embedding: Image embedding vector (vector_size,)
            image_path: Path to image file
            metadata: Additional metadata (category, defect type, etc.)
            point_id: Optional fixed ID (auto-generated if None)
            
        Returns:
            Point ID
        """
        # Use hash of path as ID if not provided
        if point_id is None:
            point_id = abs(hash(image_path)) % (10 ** 8)
        
        # Prepare point
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "image_path": str(image_path),
                "category": metadata.get("category", "unknown") if metadata else "unknown",
                "defect_type": metadata.get("defect_type", "none") if metadata else "none",
                "is_good": metadata.get("is_good", False) if metadata else False,
                **(metadata or {})
            }
        )
        
        # Upsert to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        # Store metadata locally
        self.metadata_store[point_id] = {
            "image_path": str(image_path),
            **(metadata or {})
        }
        
        return point_id
    
    def add_images_batch(self, embeddings: np.ndarray,
                        image_paths: List[str],
                        metadata_list: List[Dict] = None) -> List[int]:
        """
        Add multiple image embeddings
        
        Args:
            embeddings: Embedding matrix (N, vector_size)
            image_paths: List of image paths
            metadata_list: List of metadata dicts
            
        Returns:
            List of point IDs
        """
        if metadata_list is None:
            metadata_list = [{}] * len(image_paths)
        
        points = []
        point_ids = []
        
        for i, (embedding, image_path, metadata) in enumerate(
            zip(embeddings, image_paths, metadata_list)
        ):
            point_id = abs(hash(image_path)) % (10 ** 8)
            point_ids.append(point_id)
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "image_path": str(image_path),
                    "category": metadata.get("category", "unknown"),
                    "defect_type": metadata.get("defect_type", "none"),
                    "is_good": metadata.get("is_good", False),
                    **(metadata or {})
                }
            )
            points.append(point)
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"OK Added {len(points)} images to Qdrant")
        return point_ids
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 5,
               score_threshold: float = 0.0,
               filters: Dict = None,
               vector_name: str = None) -> List[Dict]:
        """
        Search for similar points with Qdrant using a version-safe API.
        """
        query_kwargs = {
            "collection_name": self.collection_name,
            "limit": top_k,
            "score_threshold": score_threshold,
        }
        if filters is not None:
            query_kwargs["query_filter"] = filters
        if vector_name is not None:
            # Qdrant v1 uses `using` for named vector fields, older versions may use `vector_name`.
            query_kwargs["using"] = vector_name
            query_kwargs["vector_name"] = vector_name

        query_value = query_embedding.tolist()

        results = None
        for method_name in ["search", "query", "query_points", "search_points"]:
            if hasattr(self.client, method_name):
                try:
                    method_kwargs = query_kwargs.copy()
                    if method_name == "search":
                        method_kwargs["query_vector"] = query_value
                    else:
                        method_kwargs["query"] = query_value
                    results = getattr(self.client, method_name)(**method_kwargs)
                    break
                except TypeError:
                    continue
                except Exception as exc:
                    raise RuntimeError(f"Qdrant search failed using {method_name}: {exc}")

        if results is None:
            raise RuntimeError("No supported search method found on Qdrant client")

        output = []
        for result in results:
            payload = getattr(result, "payload", {}) or {}
            output.append({
                "image_path": payload.get("image_path", ""),
                "similarity": getattr(result, "score", None) or getattr(result, "payload", {}).get("score", 0.0),
                "defect_type": payload.get("defect_type", "none"),
                "category": payload.get("category", "unknown"),
                "is_good": payload.get("is_good", False),
                "point_id": getattr(result, "id", None)
            })
        
        return output
    
    def search_by_category(self, query_embedding: np.ndarray,
                          category: str,
                          top_k: int = 5) -> List[Dict]:
        """
        Search within a specific product category
        
        Args:
            query_embedding: Query vector
            category: Product category (bottle, cable, etc.)
            top_k: Number of results
            
        Returns:
            List of similar images in category
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters={
                "must": [
                    {
                        "key": "category",
                        "match": {"value": category}
                    }
                ]
            }
        )

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vector_count": info.points_count,
            "vector_size": self.vector_size,
            "status": info.status
        }

    def clear(self):
        """Clear all data in collection"""
        self.client.delete_collection(self.collection_name)
        self._init_collection()
        self.metadata_store = {}
        print(f"OK Cleared collection: {self.collection_name}")
    
    def export_metadata(self, path: str):
        """Export metadata to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.metadata_store, f, indent=2)
        
        print(f"✓ Metadata exported to {path}")


if __name__ == "__main__":
    # Test Qdrant setup
    print("Testing Qdrant Vector Database...\n")
    
    # Initialize in-memory database
    db = QdrantVectorDB(
        collection_name="sentinel-test",
        vector_size=512,
        in_memory=True
    )
    
    # Add dummy embeddings
    print("Adding test embeddings...")
    embeddings = np.random.randn(10, 512).astype(np.float32)
    paths = [f"test_image_{i}.png" for i in range(10)]
    metadata = [
        {"category": "bottle", "defect_type": "crack", "is_good": False},
        {"category": "bottle", "defect_type": "none", "is_good": True},
    ] * 5
    
    db.add_images_batch(embeddings, paths, metadata)
    
    # Test search
    print("Testing search...")
    query = embeddings[0]
    results = db.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['image_path']} (sim: {result['similarity']:.3f})")
    
    # Get stats
    print("\nDatabase stats:")
    stats = db.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nOK Qdrant integration ready for production")
