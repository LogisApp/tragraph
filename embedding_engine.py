import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


class EmbeddingEngine:
    """
    Motor de embeddings visuales usando CLIP + FAISS.
    Genera embeddings de imágenes y permite búsqueda por similitud.
    """

    INDEX_PATH = "data/faiss_index.bin"
    METADATA_PATH = "data/metadata.json"
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model.eval()

        self.index = None
        self.metadata = []

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Genera un embedding 512d para una imagen.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            features = outputs.pooler_output
            # Proyectar al espacio compartido CLIP
            features = self.model.visual_projection(features)

        # Normalizar para similitud coseno
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().flatten()

    def build_index(self, frames_dict: dict):
        """
        Construye índice FAISS a partir de un dict {video_name: [frame_paths]}.
        Persiste index y metadata en disco.
        """
        all_vectors = []
        self.metadata = []

        total_frames = sum(len(paths) for paths in frames_dict.values())
        processed = 0

        for video_name, frame_paths in frames_dict.items():
            for frame_path in frame_paths:
                vector = self.encode_image(frame_path)
                all_vectors.append(vector)
                self.metadata.append({
                    "video": video_name,
                    "frame": os.path.basename(frame_path),
                    "path": frame_path,
                })
                processed += 1
                if processed % 10 == 0:
                    print(f"  Embeddings: {processed}/{total_frames}")

        if not all_vectors:
            raise ValueError("No se encontraron frames para indexar.")

        # Crear índice FAISS con similitud coseno (Inner Product sobre vectores normalizados)
        vectors = np.array(all_vectors, dtype=np.float32)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        # Persistir
        self._save()
        print(f"  Index creado: {len(all_vectors)} vectores, dimensión {dimension}")

    def query(self, image_path: str, top_k: int = 5) -> list:
        """
        Busca las imágenes más similares al gráfico dado.
        Retorna lista de {video, frame, score}.
        """
        if self.index is None:
            self._load()

        query_vector = self.encode_image(image_path).reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def get_best_strategy(self, image_path: str, top_k: int = 10) -> dict:
        """
        Determina la estrategia más probable por votación de los top-K matches.
        Retorna {strategy, confidence, matches}.
        """
        matches = self.query(image_path, top_k)

        if not matches:
            return {"strategy": "unknown", "confidence": 0, "matches": []}

        # Votación ponderada por score
        strategy_scores = {}
        for match in matches:
            video = match["video"]
            score = match["score"]
            strategy_scores[video] = strategy_scores.get(video, 0) + score

        best_strategy = max(strategy_scores, key=strategy_scores.get)
        total_score = sum(strategy_scores.values())
        confidence = (strategy_scores[best_strategy] / total_score) * 100 if total_score > 0 else 0

        return {
            "strategy": best_strategy,
            "confidence": round(confidence, 1),
            "matches": matches,
        }

    def _save(self):
        """Persiste index y metadata a disco."""
        os.makedirs(os.path.dirname(self.INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, self.INDEX_PATH)
        with open(self.METADATA_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load(self):
        """Carga index y metadata desde disco."""
        if not os.path.exists(self.INDEX_PATH):
            raise FileNotFoundError("No se encontró el índice FAISS. Ejecuta el entrenamiento primero.")
        self.index = faiss.read_index(self.INDEX_PATH)
        with open(self.METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    def is_trained(self) -> bool:
        """Verifica si existe un índice entrenado."""
        return os.path.exists(self.INDEX_PATH) and os.path.exists(self.METADATA_PATH)
