from pydantic import BaseModel, Field
from typing import List, Dict


class TradeLevels(BaseModel):
    entry: float = Field(..., description="Nivel de precio sugerido de entrada")
    stop_loss: float = Field(..., description="Nivel de Stop Loss")
    take_profit: float = Field(..., description="Nivel de Take Profit")


class StrategyResponse(BaseModel):
    estrategia_recomendada: str
    confianza: float
    puntos_de_entrada_salida: TradeLevels
    razonamiento_tecnico: str
