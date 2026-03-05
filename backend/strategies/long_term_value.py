"""
AutoML_Quant_Trade - 장기 가치투자 전략 (6단계 포트폴리오 기반)

AssetUniverseMapper의 6단계 리스크 프로필을 바탕으로
지정된 리밸런싱 주기마다 타겟 비중을 달성하기 위한 시그널 생성.
"""
from typing import Optional, Dict

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy
from backend.data.asset_universe import AssetUniverseMapper


class LongTermValueStrategy(BaseStrategy):
    """6단계 포트폴리오 기반 장기 가치투자 전략"""

    def __init__(self, profile: str = "Balanced",
                 rebalance_freq: int = 21,
                 rebalance_threshold: float = 0.03):
        super().__init__(name="LongTermValue")
        self.profile = profile
        self.rebalance_freq = rebalance_freq
        self.rebalance_threshold = rebalance_threshold
        
        self.mapper = AssetUniverseMapper()
        # {ticker: target_weight}
        self.target_weights = self._calculate_target_weights()
        
        # ticker별 경과 시간(봉) 추적
        self._bars_since_rebalance: Dict[str, int] = {}
        
    def _calculate_target_weights(self) -> Dict[str, float]:
        """자산군 비중을 구체적 종목(ticker) 비중으로 매핑"""
        allocation = self.mapper.get_target_portfolio(self.profile) if self.profile != "Custom" else {}
        resolved = self.mapper.resolve_to_codes(allocation)
        
        weights = {}
        for code, info in resolved.items():
            weights[code] = info["weight"]
            
        return weights
        
    def set_profile(self, profile: str, regime_weights: Dict[str, float] = None):
        """런타임에 국면 전환 등에 의해 프로필이 변경될 때 호출"""
        self.profile = profile
        if profile == "Custom" and regime_weights:
            allocation = self.mapper.get_target_portfolio(profile, regime_weights)
        else:
            allocation = self.mapper.get_target_portfolio(profile)
        
        resolved = self.mapper.resolve_to_codes(allocation)
        self.target_weights = {code: info["weight"] for code, info in resolved.items()}

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        self._record(event)
        
        ticker = event.ticker
        if ticker not in self._bars_since_rebalance:
            # 첫 번째 데이터 시 즉각 편입을 유도하기 위해 설정
            self._bars_since_rebalance[ticker] = self.rebalance_freq
            
        self._bars_since_rebalance[ticker] += 1
        
        # 리밸런싱 주기가 도래했거나 초과한 경우 시그널 생성
        if self._bars_since_rebalance[ticker] > self.rebalance_freq:
            self._bars_since_rebalance[ticker] = 0
            
            target_weight = self.target_weights.get(ticker, 0.0)

            # BUG-3 FIX: 타겟 비중이 0인 종목은 시그널 무시 (수수료 누적 방지)
            if target_weight == 0.0:
                return None
            
            # EventLoop에게 "TARGET" direction을 보내 목표 비중 달성을 요청
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=ticker,
                direction="TARGET",
                strength=target_weight,
                strategy_name=self.name,
            )
            
        return None

    def get_timeframe(self) -> str:
        return "weekly"
