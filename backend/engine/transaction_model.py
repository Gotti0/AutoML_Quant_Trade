"""
AutoML_Quant_Trade - 거래 비용 시뮬레이션 모델

제곱근 시장 충격 모델 + 호가 스프레드 + 수수료/세금.

■ 미래참조 편향 방지:
  체결가는 반드시 t+1 시가(Open) 기준이며,
  t 시점의 종가(Close)는 체결에 사용하지 않음.
"""
import logging
import numpy as np

from backend.config.settings import Settings
from backend.engine.events import OrderEvent, FillEvent, MarketEvent

logger = logging.getLogger(__name__)


from typing import Optional

class TransactionModel:
    """거래 비용 시뮬레이션 모델 (제곱근 시장 충격)"""

    def __init__(self,
                 commission_rate: float = None,
                 tax_rate: float = None,
                 market_impact_gamma: float = None,
                 bid_ask_spread: float = None):
        """
        Parameters:
            commission_rate: 매매 수수료율 (양방향)
            tax_rate: 매도 거래세율
            market_impact_gamma: 시장 충격 계수 (제곱근 모델)
            bid_ask_spread: 평균 호가 스프레드
        """
        self.commission_rate = commission_rate or Settings.COMMISSION_RATE
        self.tax_rate = tax_rate or Settings.TAX_RATE
        self.gamma = market_impact_gamma or Settings.MARKET_IMPACT_GAMMA
        self.spread = bid_ask_spread or Settings.BID_ASK_SPREAD

    def simulate_fill(self, order: OrderEvent,
                      next_market: MarketEvent) -> Optional[FillEvent]:
        """
        주문을 t+1 봉의 시가로 체결 시뮬레이션. 
        부분 체결(거래량 한도) 및 슬리피지 제한 적용.
        """
        base_price = next_market.open  # t+1 시가 기준
        if base_price <= 0:
            return None

        # 일평균 거래량을 해당 이벤트 데이터로 근사
        adv = max(next_market.volume, 1)

        # 최대 참여율(participation rate)을 해당 캔들 거래량의 10%로 물리적 제한
        max_executable_qty = max(1, int(adv * 0.10))
        executed_qty = min(abs(order.qty), max_executable_qty)

        if executed_qty <= 0:
            return None

        # 1. 호가 스프레드 비용 (절반)
        spread_cost = base_price * (self.spread / 2)

        # 2. 시장 충격 (제곱근 모델: γ × √(quantity / ADV))
        participation_rate = executed_qty / adv
        impact = self.gamma * np.sqrt(participation_rate)
        
        # 슬리피지 최대 3% 로 안전 모루 제한 (비정상 호가 변동 방어)
        impact = min(impact, 0.03)
        impact_cost = base_price * impact

        # 3. 슬리피지 방향 적용
        total_slippage = spread_cost + impact_cost
        if order.side == "BUY":
            execution_price = base_price + total_slippage  # 매수: 비싸게
        else:
            execution_price = base_price - total_slippage  # 매도: 싸게

        # 4. 수수료 계산 (양방향)
        trade_value = executed_qty * execution_price
        commission = trade_value * self.commission_rate

        # 5. 매도 거래세 (매도만)
        tax = 0.0
        if order.side == "SELL":
            tax = trade_value * self.tax_rate

        total_fee = commission + tax

        fill = FillEvent(
            timestamp=next_market.timestamp,
            ticker=order.ticker,
            qty=executed_qty if order.side == "BUY" else -executed_qty,
            execution_price=execution_price,
            fee=total_fee,
            slippage=total_slippage,
        )

        return fill
