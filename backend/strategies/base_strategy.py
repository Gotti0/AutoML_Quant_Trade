"""
AutoML_Quant_Trade - 전략 기본 인터페이스

모든 전략은 이 ABC를 상속하여 구현.
"""
from abc import ABC, abstractmethod
from typing import Optional

from backend.engine.events import MarketEvent, SignalEvent


class BaseStrategy(ABC):
    """전략 기본 인터페이스"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._history: list = []  # 과거 MarketEvent 축적

    @abstractmethod
    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        """
        새 MarketEvent를 수신하고 매매 시그널을 생성.

        ■ 미래참조 편향 방지:
          - event와 self._history만 참조 가능
          - 미래 데이터는 아키텍처상 접근 불가

        Parameters:
            event: 현재 봉의 시장 데이터
        Returns:
            SignalEvent 또는 None (시그널 없음)
        """
        ...

    @abstractmethod
    def get_timeframe(self) -> str:
        """전략의 기본 시간 주기 반환: "minute" | "daily" | "weekly" """
        ...

    def _record(self, event: MarketEvent):
        """내부 히스토리에 이벤트 추가."""
        self._history.append(event)

    def get_close_series(self) -> list:
        """축적된 종가 리스트 반환."""
        return [e.close for e in self._history]

    def get_volume_series(self) -> list:
        """축적된 거래량 리스트 반환."""
        return [e.volume for e in self._history]

    def reset(self):
        """히스토리 초기화."""
        self._history = []
