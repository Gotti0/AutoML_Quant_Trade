"""
AutoML_Quant_Trade - 클러스터 모멘텀 전략

종목 군집 내 상대 강도(relative strength) 기반 매매.
군집 내 상위 N% → 롱, 하위 종목(보유 시) → 청산.

■ 리밸런싱 주기마다 군집 내 순위를 재계산하여
  상위 모멘텀 종목에 집중 투자.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd

from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ClusterMomentumStrategy(BaseStrategy):
    """군집 내 상대 강도 기반 롱 전략"""

    def __init__(self, cluster_analyzer,
                 top_pct: float = 0.2,
                 rebalance_freq: int = 21,
                 target_clusters: Optional[List[int]] = None):
        """
        Parameters:
            cluster_analyzer: CrossAssetClusterAnalyzer 인스턴스
            top_pct: 군집 내 상위 비율 (0~1)
            rebalance_freq: 리밸런싱 주기 (거래일)
            target_clusters: 투자 대상 군집 ID 리스트 (None=전체)
        """
        super().__init__()
        self.cluster_analyzer = cluster_analyzer
        self.top_pct = top_pct
        self.rebalance_freq = rebalance_freq
        self.target_clusters = target_clusters

        self._day_counter: dict = {}
        self._selected_tickers: List[str] = []

    def get_timeframe(self) -> str:
        return "daily"

    def set_target_clusters(self, cluster_ids: List[int]):
        """투자 대상 군집을 동적으로 설정 (스크리너 연동)."""
        self.target_clusters = cluster_ids

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        """시장 데이터 처리 → 군집 내 순위 기반 시그널."""
        self._record(event)
        ticker = event.ticker

        if ticker not in self._day_counter:
            self._day_counter[ticker] = 0
        self._day_counter[ticker] += 1

        # 리밸런싱 주기마다 선정 종목 갱신
        if self._day_counter[ticker] % self.rebalance_freq == 0:
            self._update_selection()

        # 선정 종목에 포함 → BUY, 미포함(기존 보유) → SELL
        if ticker in self._selected_tickers:
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=ticker,
                direction="TARGET",
                strength=1.0 / max(len(self._selected_tickers), 1),
            )

        return None

    def _update_selection(self):
        """군집 내 상위 모멘텀 종목 리스트 갱신."""
        try:
            clusters = self.cluster_analyzer.get_clusters()
            if not clusters:
                return

            # 대상 군집 결정
            target_ids = self.target_clusters
            if target_ids is None:
                target_ids = [k for k in clusters.keys() if k != -1]

            all_ranked = []

            for cluster_id in target_ids:
                ranked = self.cluster_analyzer.rank_within_cluster(cluster_id)
                if ranked.empty:
                    continue

                # 상위 top_pct 선정
                n_select = max(1, int(len(ranked) * self.top_pct))
                top_tickers = ranked.head(n_select)["ticker"].tolist()
                all_ranked.extend(top_tickers)

            self._selected_tickers = all_ranked
            logger.debug(
                f"ClusterMomentum: {len(self._selected_tickers)} tickers selected "
                f"from {len(target_ids)} clusters"
            )

        except Exception as e:
            logger.error(f"ClusterMomentum selection failed: {e}")
