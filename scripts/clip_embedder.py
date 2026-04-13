"""
CLIP Embedding Generator
Generates embeddings for images and text for multimodal similarity search
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Union
import cv2
from PIL import Image
import open_clip


class CLIPEmbedder:
    """
    Generate and manage CLIP embeddings for images and text
    Enables multimodal search: similar images, semantic matching, etc.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        """
        Initialize CLIP model
        
        Args:
            model_name: Model architecture (ViT-B-32, ViT-L-14, etc.)
            pretrained: Pretrained weights (openai, laion400M, etc.)
            device: torch device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.pretrained = pretrained
        
        print(f"Loading CLIP model {model_name} ({pretrained}) on {self.device}...")
        
        # Load model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Embedding dimension
        self.embed_dim = self.model.visual.output_dim if hasattr(self.model, 'visual') else 512
    
    def embed_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image: Image path, numpy array (H, W, 3), or PIL Image
            
        Returns:
            Embedding vector (embed_dim,), L2 normalized
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert BGR (from cv2) to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Preprocess
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
        
        return embedding.squeeze(0).cpu().numpy()
    
    def embed_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            Embedding matrix (N, embed_dim)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        if embeddings:
            return np.vstack(embeddings)
        return np.array([])
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embedding for text query
        
        Args:
            text: Text string or list of strings
            
        Returns:
            Embedding vector or matrix (1/N, embed_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        text_tokens = self.tokenizer(text).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        result = text_embeddings.cpu().numpy()
        return result.squeeze(0) if len(text) == 1 else result
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1, embedding2: Embedding vectors
            
        Returns:
            Similarity score [-1, 1], 1 = identical
        """
        # Ensure normalized
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        return float(np.dot(e1, e2))
    
    def find_similar_images(self, query_embedding: np.ndarray, 
                           database_embeddings: np.ndarray, 
                           top_k: int = 5) -> np.ndarray:
        """
        Find top-k similar images from database
        
        Args:
            query_embedding: Query image embedding (embed_dim,)
            database_embeddings: Database embeddings (N, embed_dim)
            top_k: Number of results
            
        Returns:
            Indices of top-k similar images (sorted by similarity, highest first)
        """
        # Normalize query
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(database_embeddings, query_embedding)
        
        # Get top-k
        topk_indices = np.argsort(similarities)[::-1][:top_k]
        
        return topk_indices


if __name__ == "__main__":
    # Test CLIP embeddings
    mvtec_root = Path("../data/mvtec/bottle")
    
    embedder = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
    
    # Get some images
    good_images = list((mvtec_root / "test" / "good").glob("*.png"))[:5]
    defect_images = list((mvtec_root / "test" / "broken_large").glob("*.png"))[:5]
    
    print("\nTesting image embeddings:")
    if good_images:
        good_embedding = embedder.embed_image(str(good_images[0]))
        print(f"✓ Image embedding shape: {good_embedding.shape}")
    
    print("\nTesting text embeddings:")
    texts = [
        "a defective bottle with cracks",
        "a good bottle without defects",
        "bottle damage"
    ]
    text_embeddings = embedder.embed_text(texts)
    print(f"✓ Text embeddings shape: {text_embeddings.shape}")
    
    print("\nTesting similarity search:")
    if good_images:
        all_images = [str(p) for p in good_images + defect_images]
        embeddings = embedder.embed_images(all_images)
        
        query = embedder.embed_image(str(good_images[0]))
        similar_idx = embedder.find_similar_images(query, embeddings, top_k=3)
        
        print(f"Query: {good_images[0].name}")
        for idx in similar_idx:
            print(f"  → {Path(all_images[idx]).relative_to(mvtec_root)}")
    
    print("\n✓ CLIP embedder ready for production")
