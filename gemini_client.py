import os
from google import genai
from google.genai import types
from utils import validate_video_size
from base_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    Cliente encapsulado para interacción con Gemini multimodal.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name

    def upload_file(self, file_path: str):
        """
        Carga un archivo (imagen o video) a Gemini.
        """
        if file_path.endswith(".mp4"):
            validate_video_size(file_path)

        return self.client.files.upload(file=file_path)

    def analyze_multimodal(self, chart_file, strategy_videos):
        """
        Envía gráfico + videos como contexto para análisis comparativo.
        """

        prompt = """
        Analiza el gráfico de trading proporcionado (velas japonesas 1H).
        Identifica:
        - Tendencia actual
        - Soportes y resistencias
        - Condición de RSI si es visible
        - Posibles patrones técnicos

        Luego compara visualmente el gráfico con los videos de referencia.
        Determina cuál estrategia coincide más (ej. breakout, mean_reversion).

        Devuelve EXCLUSIVAMENTE un JSON con la siguiente estructura:

        {
          "estrategia_recomendada": "...",
          "confianza": 0-100,
          "puntos_de_entrada_salida": {
              "entry": float,
              "stop_loss": float,
              "take_profit": float
          },
          "razonamiento_tecnico": "Explicación técnica breve"
        }
        """

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_uri(file_uri=chart_file.uri, mime_type=chart_file.mime_type),
                    *[types.Part.from_uri(file_uri=video.uri, mime_type=video.mime_type) for video in strategy_videos],
                ],
            )
        ]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        return response.text

    def generate_reasoning(self, chart_path: str, strategy: str, confidence: float):
        """
        Genera razonamiento técnico basado en el gráfico + estrategia identificada.
        Solo envía 1 imagen + texto (sin videos), reduciendo costo ~10x.
        """
        chart_file = self.upload_file(chart_path)

        prompt = f"""
        Se ha identificado el patrón "{strategy}" con {confidence:.1f}% de confianza
        mediante análisis vectorial del siguiente gráfico de trading.

        Analiza el gráfico proporcionado y:
        1. Valida si el patrón "{strategy}" es coherente con lo que ves
        2. Identifica niveles técnicos clave (soportes, resistencias)
        3. Sugiere puntos de entrada, stop loss y take profit

        Devuelve EXCLUSIVAMENTE un JSON con esta estructura:

        {{
          "estrategia_recomendada": "{strategy}",
          "confianza": 0-100,
          "puntos_de_entrada_salida": {{
              "entry": float,
              "stop_loss": float,
              "take_profit": float
          }},
          "razonamiento_tecnico": "Explicación técnica breve"
        }}
        """

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_uri(
                        file_uri=chart_file.uri, mime_type=chart_file.mime_type
                    ),
                ],
            )
        ]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        return response.text

