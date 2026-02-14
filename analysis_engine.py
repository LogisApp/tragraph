import json
import os
from dataclasses import dataclass
from embedding_engine import EmbeddingEngine
from extrae_video import extract_all_videos
from quant_engine import QuantEngine, QuantValidation
from risk_engine import RiskEngine, RiskConfig, RiskDecision
from schemas import StrategyResponse


# Registry de proveedores disponibles
LLM_PROVIDERS = {
    "Gemini 2.0 Flash": {"module": "gemini_client", "class": "GeminiClient", "key_env": "GOOGLE_API_KEY"},
    "GPT-4o (OpenAI)": {"module": "openai_client", "class": "OpenAIClient", "key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
    "GPT-4 Turbo (OpenAI)": {"module": "openai_client", "class": "OpenAIClient", "key_env": "OPENAI_API_KEY", "model": "gpt-4-turbo"},
    "Llama Vision (Ollama)": {"module": "llama_client", "class": "LlamaClient", "key_env": None, "model": "llama3.2-vision"},
}


def get_llm_client(provider_name: str):
    """Factory: crea el cliente LLM según el proveedor seleccionado."""
    import importlib
    provider = LLM_PROVIDERS.get(provider_name)
    if not provider:
        raise ValueError(f"Proveedor no válido: {provider_name}")
    module = importlib.import_module(provider["module"])
    client_class = getattr(module, provider["class"])
    model = provider.get("model")
    return client_class(model_name=model) if model else client_class()


@dataclass
class PipelineResult:
    """Resultado completo del pipeline de 4 etapas."""
    # Etapa 1: Clasificación (CLIP + FAISS)
    strategy: str
    clip_confidence: float
    matches: list

    # Etapa 2: Razonamiento LLM
    llm_response: StrategyResponse

    # Etapa 3: Validación Cuantitativa
    quant: QuantValidation

    # Etapa 4: Decisión de Riesgo
    risk: RiskDecision

    # Decisión final
    should_execute: bool
    rejection_reason: str = ""


class TradingAnalysisEngine:

    VIDEOS_DIR = "videos"
    FRAMES_DIR = "data/frames"

    def __init__(self, risk_config: RiskConfig = None):
        self.embeddings = EmbeddingEngine()
        self.quant = QuantEngine()
        self.risk = RiskEngine(risk_config)

    def entrenar(self, videos_dir=None, frame_interval=30, progress_callback=None):
        """
        Pipeline de entrenamiento:
        1. Extrae keyframes de todos los videos
        2. Genera embeddings CLIP
        3. Construye índice FAISS
        """
        videos_dir = videos_dir or self.VIDEOS_DIR

        if progress_callback:
            progress_callback("Extrayendo keyframes de videos...")

        frames_dict = extract_all_videos(
            videos_dir, self.FRAMES_DIR, frame_interval
        )

        total_frames = sum(len(v) for v in frames_dict.values())

        if progress_callback:
            progress_callback(f"Generando embeddings para {total_frames} frames...")

        self.embeddings.build_index(frames_dict)

        return {
            "videos_procesados": len(frames_dict),
            "total_frames": total_frames,
            "videos": {k: len(v) for k, v in frames_dict.items()}
        }

    def analizar_pipeline(self, chart_path: str,
                          provider_name: str = "Gemini 2.0 Flash",
                          symbol: str = "EURUSD") -> PipelineResult:
        """
        Pipeline institucional de 4 etapas:
        1. CLIP + FAISS → Clasificación de patrón
        2. LLM → Razonamiento técnico (solo contexto)
        3. Quant Engine → Validación matemática
        4. Risk Engine → Control de exposición

        El LLM NO decide. Solo clasifica y contextualiza.
        """

        # ═══════════════════════════════════════════
        # ETAPA 1: Clasificación (CLIP + FAISS)
        # ═══════════════════════════════════════════
        match_result = self.embeddings.get_best_strategy(chart_path)
        strategy = match_result["strategy"]
        clip_confidence = match_result["confidence"]

        # ═══════════════════════════════════════════
        # ETAPA 2: Razonamiento LLM (solo contexto)
        # ═══════════════════════════════════════════
        llm = get_llm_client(provider_name)
        reasoning_response = llm.generate_reasoning(chart_path, strategy, clip_confidence)
        data = json.loads(reasoning_response)

        if "confianza" not in data or data["confianza"] == 0:
            data["confianza"] = clip_confidence
        if "estrategia_recomendada" not in data or not data["estrategia_recomendada"]:
            data["estrategia_recomendada"] = strategy

        llm_response = StrategyResponse(**data)

        entry = llm_response.puntos_de_entrada_salida.entry
        sl = llm_response.puntos_de_entrada_salida.stop_loss
        tp = llm_response.puntos_de_entrada_salida.take_profit

        # ═══════════════════════════════════════════
        # ETAPA 3: Validación Cuantitativa
        # ═══════════════════════════════════════════
        quant_result = self.quant.validate(
            strategy=strategy, entry=entry,
            stop_loss=sl, take_profit=tp,
            confidence=clip_confidence,
        )

        # ═══════════════════════════════════════════
        # ETAPA 4: Risk Engine
        # ═══════════════════════════════════════════
        risk_result = self.risk.evaluate(
            strategy=strategy, entry=entry,
            stop_loss=sl, take_profit=tp,
            quant_score=quant_result.quant_score,
            symbol=symbol,
        )

        # ═══════════════════════════════════════════
        # DECISIÓN FINAL
        # ═══════════════════════════════════════════
        should_execute = quant_result.approved and risk_result.approved
        rejection_reason = ""

        if not quant_result.approved:
            rejection_reason = "Rechazado por Quant Engine"
        elif not risk_result.approved:
            rejection_reason = "Rechazado por Risk Engine"

        return PipelineResult(
            strategy=strategy,
            clip_confidence=clip_confidence,
            matches=match_result.get("matches", []),
            llm_response=llm_response,
            quant=quant_result,
            risk=risk_result,
            should_execute=should_execute,
            rejection_reason=rejection_reason,
        )

    def ejecutar_si_valido(self, pipeline_result: PipelineResult, symbol="EURUSD"):
        """
        Ejecuta SOLO si el pipeline completo aprobó.
        """
        from mt5_executor import MT5Executor, MT5Config

        if not pipeline_result.should_execute:
            return f"⛔ Operación rechazada: {pipeline_result.rejection_reason}"

        levels = pipeline_result.llm_response.puntos_de_entrada_salida
        direction = "BUY" if levels.stop_loss < levels.entry else "SELL"

        config = MT5Config(symbol=symbol, risk_percent=pipeline_result.risk.risk_percent)
        executor = MT5Executor(config)

        try:
            result = executor.send_market_order(
                direction=direction,
                entry=levels.entry,
                stop_loss=levels.stop_loss,
                take_profit=levels.take_profit,
            )
            return result
        finally:
            executor.shutdown()
