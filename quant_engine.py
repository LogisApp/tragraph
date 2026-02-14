"""
Motor Cuantitativo — Validación matemática de patrones.
No depende del LLM: valida con reglas determinísticas.
"""

from dataclasses import dataclass


@dataclass
class QuantValidation:
    """Resultado de la validación cuantitativa."""
    quant_score: float          # 0-100
    approved: bool              # True si score >= threshold
    range_compression: float    # Ratio rango actual vs promedio
    risk_reward_ratio: float    # R:R calculado
    breakout_confirmed: bool    # Si la ruptura es coherente
    pattern_valid: bool         # Si el patrón es matemáticamente válido
    details: str                # Explicación


class QuantEngine:
    """
    Valida matemáticamente si los niveles propuestos por el LLM
    tienen fundamento técnico real.
    """

    SCORE_THRESHOLD = 50  # Mínimo para aprobar

    def validate(self, strategy: str, entry: float, stop_loss: float,
                 take_profit: float, confidence: float) -> QuantValidation:
        """
        Validación cuantitativa basada en los niveles propuestos.
        """
        details = []
        score = 0

        # --- 1. Validar coherencia de niveles ---
        levels_valid = self._validate_levels(entry, stop_loss, take_profit, strategy)
        if levels_valid:
            score += 25
            details.append("✅ Niveles coherentes con dirección")
        else:
            details.append("❌ Niveles incoherentes con la estrategia")

        # --- 2. Risk/Reward Ratio ---
        rr_ratio = self._calculate_rr(entry, stop_loss, take_profit)
        if rr_ratio >= 2.0:
            score += 25
            details.append(f"✅ R:R = {rr_ratio:.2f} (≥ 2.0)")
        elif rr_ratio >= 1.5:
            score += 15
            details.append(f"⚠️ R:R = {rr_ratio:.2f} (aceptable, pero < 2.0)")
        else:
            details.append(f"❌ R:R = {rr_ratio:.2f} (< 1.5, desfavorable)")

        # --- 3. Stop Loss no excesivo ---
        sl_distance_pct = abs(entry - stop_loss) / entry * 100
        if sl_distance_pct <= 2.0:
            score += 20
            details.append(f"✅ SL distance = {sl_distance_pct:.2f}% (controlado)")
        elif sl_distance_pct <= 5.0:
            score += 10
            details.append(f"⚠️ SL distance = {sl_distance_pct:.2f}% (amplio)")
        else:
            details.append(f"❌ SL distance = {sl_distance_pct:.2f}% (excesivo)")

        # --- 4. Compresión de rango ---
        range_compression = abs(take_profit - stop_loss) / entry * 100
        if range_compression > 0.5:
            score += 15
            details.append(f"✅ Rango operativo = {range_compression:.2f}%")
        else:
            details.append(f"❌ Rango demasiado estrecho = {range_compression:.2f}%")

        # --- 5. Confianza del clasificador ---
        if confidence >= 70:
            score += 15
            details.append(f"✅ Confianza CLIP = {confidence:.1f}%")
        elif confidence >= 50:
            score += 8
            details.append(f"⚠️ Confianza CLIP = {confidence:.1f}% (baja)")
        else:
            details.append(f"❌ Confianza CLIP = {confidence:.1f}% (insuficiente)")

        # --- Patrón válido ---
        pattern_valid = levels_valid and rr_ratio >= 1.5
        breakout_confirmed = levels_valid and rr_ratio >= 2.0 and confidence >= 60

        approved = score >= self.SCORE_THRESHOLD

        return QuantValidation(
            quant_score=min(score, 100),
            approved=approved,
            range_compression=range_compression,
            risk_reward_ratio=rr_ratio,
            breakout_confirmed=breakout_confirmed,
            pattern_valid=pattern_valid,
            details="\n".join(details),
        )

    def _validate_levels(self, entry, sl, tp, strategy):
        """Verifica que SL/TP sean coherentes con la dirección."""
        strategy_lower = strategy.lower()

        if any(kw in strategy_lower for kw in ["buy", "long", "breakout", "trend_up"]):
            return sl < entry < tp
        elif any(kw in strategy_lower for kw in ["sell", "short", "mean_rev"]):
            return tp < entry < sl
        else:
            # Inferir dirección desde niveles
            return (sl < entry < tp) or (tp < entry < sl)

    def _calculate_rr(self, entry, sl, tp):
        """Calcula el Risk/Reward ratio."""
        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return 0.0

        return reward / risk
