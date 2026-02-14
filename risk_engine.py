"""
Risk Engine — Control de exposición y validación de riesgo.
Última barrera antes de la ejecución.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date


RISK_STATE_PATH = "data/risk_state.json"


@dataclass
class RiskDecision:
    """Resultado de la evaluación de riesgo."""
    approved: bool
    position_size: float        # Lotes recomendados
    risk_amount: float          # $ en riesgo
    risk_percent: float         # % de cuenta en riesgo
    daily_loss_remaining: float # Pérdida diaria restante
    details: str                # Explicación


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 1.0    # % máximo por operación
    max_daily_loss: float = 3.0         # % máximo pérdida diaria
    min_rr_ratio: float = 2.0           # R:R mínimo requerido
    max_open_positions: int = 3         # Máximo posiciones simultáneas
    max_correlation_exposure: int = 2   # Máx posiciones en mismo par/grupo
    account_balance: float = 10000.0    # Balance de cuenta (editable)


class RiskEngine:
    """
    Controla exposición, position sizing y límites de riesgo.
    Mantiene estado diario persistente.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.state = self._load_state()

    def evaluate(self, strategy: str, entry: float, stop_loss: float,
                 take_profit: float, quant_score: float,
                 symbol: str = "EURUSD") -> RiskDecision:
        """
        Evaluación completa de riesgo antes de ejecutar.
        """
        details = []

        # --- 1. Verificar quant_score ---
        if quant_score < 50:
            return RiskDecision(
                approved=False, position_size=0, risk_amount=0,
                risk_percent=0, daily_loss_remaining=self._daily_loss_remaining(),
                details="❌ Quant score insuficiente. No se evalúa riesgo.",
            )

        # --- 2. R:R Ratio ---
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr = reward / risk if risk > 0 else 0

        if rr < self.config.min_rr_ratio:
            return RiskDecision(
                approved=False, position_size=0, risk_amount=0,
                risk_percent=0, daily_loss_remaining=self._daily_loss_remaining(),
                details=f"❌ R:R = {rr:.2f} < mínimo {self.config.min_rr_ratio}",
            )
        details.append(f"✅ R:R = {rr:.2f}")

        # --- 3. Pérdida diaria ---
        daily_remaining = self._daily_loss_remaining()
        if daily_remaining <= 0:
            return RiskDecision(
                approved=False, position_size=0, risk_amount=0,
                risk_percent=0, daily_loss_remaining=0,
                details="❌ Límite de pérdida diaria alcanzado. No operar más hoy.",
            )
        details.append(f"✅ Pérdida diaria restante: ${daily_remaining:.2f}")

        # --- 4. Posiciones abiertas ---
        open_positions = self.state.get("open_positions", 0)
        if open_positions >= self.config.max_open_positions:
            return RiskDecision(
                approved=False, position_size=0, risk_amount=0,
                risk_percent=0, daily_loss_remaining=daily_remaining,
                details=f"❌ Máximo de posiciones abiertas ({self.config.max_open_positions}) alcanzado.",
            )
        details.append(f"✅ Posiciones abiertas: {open_positions}/{self.config.max_open_positions}")

        # --- 5. Position sizing ---
        risk_amount = self.config.account_balance * (self.config.max_risk_per_trade / 100)
        risk_amount = min(risk_amount, daily_remaining)

        # Calcular lotes (asumiendo forex standard: 1 pip = $10 por lote)
        pip_value = 10.0  # USD por pip por lote estándar
        risk_pips = risk / 0.0001 if "JPY" not in symbol else risk / 0.01
        position_size = risk_amount / (risk_pips * pip_value) if risk_pips > 0 else 0
        position_size = round(max(0.01, min(position_size, 10.0)), 2)

        risk_percent = (risk_amount / self.config.account_balance) * 100

        details.append(f"✅ Position size: {position_size} lotes")
        details.append(f"✅ Riesgo: ${risk_amount:.2f} ({risk_percent:.1f}%)")

        return RiskDecision(
            approved=True,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            daily_loss_remaining=daily_remaining - risk_amount,
            details="\n".join(details),
        )

    def record_trade(self, profit_loss: float):
        """Registra resultado de una operación."""
        today = str(date.today())
        if self.state.get("date") != today:
            self.state = {"date": today, "daily_pnl": 0, "trades": 0, "open_positions": 0}

        self.state["daily_pnl"] += profit_loss
        self.state["trades"] += 1
        self._save_state()

    def _daily_loss_remaining(self) -> float:
        """Calcula cuánto más se puede perder hoy."""
        today = str(date.today())
        if self.state.get("date") != today:
            self.state = {"date": today, "daily_pnl": 0, "trades": 0, "open_positions": 0}

        max_daily = self.config.account_balance * (self.config.max_daily_loss / 100)
        current_loss = abs(min(0, self.state.get("daily_pnl", 0)))
        return max(0, max_daily - current_loss)

    def _load_state(self) -> dict:
        if os.path.exists(RISK_STATE_PATH):
            with open(RISK_STATE_PATH, "r") as f:
                return json.load(f)
        return {"date": str(date.today()), "daily_pnl": 0, "trades": 0, "open_positions": 0}

    def _save_state(self):
        os.makedirs(os.path.dirname(RISK_STATE_PATH), exist_ok=True)
        with open(RISK_STATE_PATH, "w") as f:
            json.dump(self.state, f, indent=2)
