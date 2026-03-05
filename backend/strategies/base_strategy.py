"""
AutoML_Quant_Trade - 전략 기본 인터페이스

모든 전략은 이 ABC를 상속하여 구현.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from backend.engine.events import MarketEvent, SignalEvent


class BaseStrategy(ABC):
    """전략 기본 인터페이스"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        # BUG-1 FIX: 종목별 히스토리 분리 저장 (기존: 단일 list)
        self._history: Dict[str, List[MarketEvent]] = {}

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
        """내부 히스토리에 종목별로 이벤트 추가."""
        if event.ticker not in self._history:
            self._history[event.ticker] = []
        self._history[event.ticker].append(event)

    def get_close_series(self, ticker: str) -> list:
        """특정 종목의 축적된 종가 리스트 반환."""
        return [e.close for e in self._history.get(ticker, [])]

    def get_volume_series(self, ticker: str) -> list:
        """특정 종목의 축적된 거래량 리스트 반환."""
        return [e.volume for e in self._history.get(ticker, [])]

    def reset(self):
        """히스토리 초기화."""
        self._history = {}
