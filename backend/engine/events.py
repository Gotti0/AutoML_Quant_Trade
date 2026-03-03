"""
AutoML_Quant_Trade - 백테스팅 이벤트 타입 정의

이벤트 기반 아키텍처의 핵심 데이터 구조.
시간순 정렬을 위해 __lt__ 를 timestamp 기반으로 구현.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=False)
class MarketEvent:
    """가격 데이터 도착 이벤트"""
    timestamp: int            # YYYYMMDD 또는 YYYYMMDDHHMM
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "daily"  # "daily" | "minute"

    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass(order=False)
class SignalEvent:
    """전략이 생성한 매매 시그널"""
    timestamp: int
    ticker: str
    direction: str            # "BUY" | "SELL"
    strength: float           # 시그널 강도 [0, 1]
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass(order=False)
class OrderEvent:
    """포트폴리오가 생성한 주문"""
    timestamp: int
    ticker: str
    qty: int                  # 수량 (음수 = 매도)
    side: str                 # "BUY" | "SELL"
    order_type: str = "MARKET"  # "MARKET" | "LIMIT"
    limit_price: Optional[float] = None

    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass(order=False)
class FillEvent:
    """
    체결 결과.

    ■ 미래참조 편향 방지:
      timestamp는 시그널 발생 시점이 아닌,
      **다음 봉(t+1)의 시가** 기준으로 설정됨.
    """
    timestamp: int            # 체결 시점 (시그널 t → 체결 t+1)
    ticker: str
    qty: int                  # 체결 수량 (음수 = 매도)
    execution_price: float    # 체결가 (t+1 시가 × 슬리피지)
    fee: float                # 수수료 + 거래세
    slippage: float           # 슬리피지 금액

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    @property
    def total_cost(self) -> float:
        """체결 총 비용 (양수 = 매수 지출, 음수 = 매도 수입)"""
        return self.qty * self.execution_price + self.fee
