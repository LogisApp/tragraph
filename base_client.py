from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Interfaz base para clientes LLM con capacidad de análisis visual.
    Todos los proveedores deben implementar generate_reasoning.
    """

    @abstractmethod
    def generate_reasoning(self, chart_path: str, strategy: str, confidence: float) -> str:
        """
        Genera razonamiento técnico basado en gráfico + estrategia.
        Retorna JSON string con StrategyResponse.
        """
        pass

    def get_prompt(self, strategy: str, confidence: float) -> str:
        """
        Prompt estándar compartido por todos los proveedores.
        """
        return f"""
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
