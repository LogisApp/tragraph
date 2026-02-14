import os
import base64
from openai import OpenAI
from base_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    Cliente para OpenAI GPT-4o y GPT-4 Vision.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _encode_image(self, image_path: str) -> str:
        """Codifica imagen a base64 para la API de OpenAI."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_reasoning(self, chart_path: str, strategy: str, confidence: float) -> str:
        prompt = self.get_prompt(strategy, confidence)
        base64_image = self._encode_image(chart_path)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1000,
        )

        return response.choices[0].message.content
