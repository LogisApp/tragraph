try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class MT5Config:
    symbol: str
    risk_percent: float = 1.0
    magic_number: int = 20250213
    deviation: int = 20


class MT5Executor:
    """
    Executor automático de órdenes en MT5
    """

    def __init__(self, config: MT5Config):
        self.config = config

        if mt5 is None:
            raise RuntimeError("MetaTrader5 no está disponible en este sistema (requiere Windows).")

        if not mt5.initialize():
            raise RuntimeError("No se pudo inicializar MT5")

        if not mt5.symbol_select(config.symbol, True):
            raise RuntimeError(f"No se pudo seleccionar el símbolo {config.symbol}")

    def shutdown(self):
        mt5.shutdown()

    def _calculate_lot_size(self, entry, stop_loss):
        """
        Cálculo dinámico de volumen basado en riesgo.
        """
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(self.config.symbol)

        if account_info is None or symbol_info is None:
            raise RuntimeError("Error obteniendo info de cuenta o símbolo")

        balance = account_info.balance
        risk_amount = balance * (self.config.risk_percent / 100)

        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size

        stop_distance = abs(entry - stop_loss)

        if stop_distance == 0:
            raise ValueError("Stop Loss inválido")

        value_per_point = tick_value / tick_size
        lot = risk_amount / (stop_distance * value_per_point)

        lot = max(symbol_info.volume_min,
                  min(lot, symbol_info.volume_max))

        return round(lot, 2)

    def send_market_order(self, direction: str, entry: float,
                          stop_loss: float, take_profit: float):
        """
        Ejecuta orden market BUY/SELL
        """

        symbol_info_tick = mt5.symbol_info_tick(self.config.symbol)
        if symbol_info_tick is None:
            raise RuntimeError("No se pudo obtener tick actual")

        price = symbol_info_tick.ask if direction == "BUY" else symbol_info_tick.bid
        lot = self._calculate_lot_size(entry, stop_loss)

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": "AI_Gemini_Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Error enviando orden: {result.retcode}")

        return result
