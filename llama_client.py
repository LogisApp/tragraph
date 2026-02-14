import os
import base64
import json
import requests
from base_client import BaseLLMClient


class LlamaClient(BaseLLMClient):
    """
    Cliente para Llama Vision via Ollama (local).
    Requiere Ollama corriendo localmente con un modelo de visión.
    """

    def __init__(self, model_name: str = "llama3.2-vision", host: str = None):
        self.model_name = model_name
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _encode_image(self, image_path: str) -> str:
        """Codifica imagen a base64 para Ollama."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_reasoning(self, chart_path: str, strategy: str, confidence: float) -> str:
        prompt = self.get_prompt(strategy, confidence)
        base64_image = self._encode_image(chart_path)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
            },
        }

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=600,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code} - {response.text}")

        result = response.json()
        return result.get("response", "{}")
