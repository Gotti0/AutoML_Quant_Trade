"""
AutoML_Quant_Trade - 비지도학습 + 기본적분석 통합 스크리너

비지도학습 클러스터링(CrossAssetClusterAnalyzer)과
기본적분석 스코어링(FundamentalScorer)을 결합하여
투자 대상 종목을 동적으로 선별.

■ 파이프라인:
  1. 전 종목 피처 추출 + 재무 데이터 조회
  2. ClusterAnalyzer로 종목 군집화
  3. FundamentalScorer로 멀티팩터 스코어 산출
  4. AnomalyDetector로 이상 종목 필터링
  5. RegimeModel로 현재 국면 판단
  6. 스코어 종합 → 국면 맞춤 종목 추천
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.models.feature_engineer import FeatureEngineer
from backend.screener.screener_result import ScreenerResult
from backend.screener.fundamental_scorer import FundamentalScorer

logger = logging.getLogger(__name__)


def _extract_single_ticker_features(ticker: str, price_df) -> 'Optional[pd.DataFrame]':
    """
    단일 종목 피처 추출 (ProcessPoolExecutor용 최상위 함수).
    
    멀티프로세싱에서 pickle 직렬화가 가능하도록 클래스 밖에 정의.
    """
    try:
        fe = FeatureEngineer()
        features = fe.extract(price_df)
        if not features.empty and len(features) >= 20:
            return features
    except Exception:
        pass
    return None

class UnsupervisedScreener:
    """비지도학습 + 기본적분석 통합 스크리너"""

    def __init__(self, db,
                 cluster_analyzer,
                 anomaly_detector,
                 regime_model,
                 fundamental_scorer: FundamentalScorer,
                 feature_engineer: Optional[FeatureEngineer] = None,
                 tech_weight: float = 0.4,
                 fund_weight: float = 0.4,
                 anomaly_penalty_weight: float = 0.2,
                 top_n: int = 30):
        """
        Parameters:
            db: DatabaseManager 인스턴스
            cluster_analyzer: CrossAssetClusterAnalyzer 인스턴스
            anomaly_detector: AnomalyDetector 인스턴스
            regime_model: RegimeHMM/RegimeGMM 인스턴스
            fundamental_scorer: FundamentalScorer 인스턴스
            feature_engineer: FeatureEngineer (None이면 자동 생성)
            tech_weight: 기술적 스코어 가중치
            fund_weight: 기본적분석 스코어 가중치
            anomaly_penalty_weight: 이상치 페널티 가중치
            top_n: 최종 추천 종목 수
        """
        self.db = db
        self.cluster_analyzer = cluster_analyzer
        self.anomaly_detector = anomaly_detector
        self.regime_model = regime_model
        self.fundamental_scorer = fundamental_scorer
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.tech_weight = tech_weight
        self.fund_weight = fund_weight
        self.anomaly_penalty_weight = anomaly_penalty_weight
        self.top_n = top_n

    def run(self, target_date: int = None,
            market_data: Optional[Dict[str, pd.DataFrame]] = None,
            tickers: Optional[List[str]] = None) -> ScreenerResult:
        """
        스크리너 실행.

        Parameters:
            target_date: 스크리닝 기준일 (YYYYMMDD)
            market_data: {ticker: price_df} — 이미 로드된 시장 데이터
            tickers: 스크리닝 대상 종목 리스트
        Returns:
            ScreenerResult
        """
        logger.info(f"스크리너 실행 시작 (target_date={target_date})")

        # ─── 1. 데이터 준비 ───
        if market_data is None:
            market_data = self._load_market_data(tickers)

        if not market_data:
            logger.warning("No market data available for screening.")
            return self._empty_result(target_date or 0)

        all_tickers = list(market_data.keys())
        logger.info(f"스크리닝 대상: {len(all_tickers)}개 종목")

        # ─── 2. 피처 추출 (멀티프로세싱 병렬화) ───
        multi_asset_features = self._extract_features_parallel(market_data)

        if len(multi_asset_features) < 10:
            logger.warning(f"Too few tickers with valid features: {len(multi_asset_features)}")
            return self._empty_result(target_date or 0)

        logger.info(f"피처 추출 완료: {len(multi_asset_features)}개 종목")

        # ─── 3. 군집화 ───
        try:
            self.cluster_analyzer.fit(multi_asset_features)
            clusters = self.cluster_analyzer.get_clusters()
            logger.info(f"군집화 완료: {self.cluster_analyzer.n_clusters}개 군집")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            clusters = {}

        # 종목별 군집 ID
        cluster_assignments = {}
        for ticker in multi_asset_features:
            cluster_assignments[ticker] = self.cluster_analyzer.get_ticker_cluster(ticker)

        # ─── 4. 기본적분석 스코어링 ───
        scored_tickers = list(multi_asset_features.keys())
        try:
            fundamentals_df = self.fundamental_scorer.fetch_fundamentals(scored_tickers)
            if not fundamentals_df.empty:
                fundamentals_df = self.fundamental_scorer.score(fundamentals_df)
                logger.info(f"기본적분석 스코어 산출: {len(fundamentals_df)}개 종목")
            else:
                fundamentals_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Fundamental scoring failed: {e}")
            fundamentals_df = pd.DataFrame()

        # ─── 5. 이상 탐지 ───
        anomaly_flags: Dict[str, bool] = {}
        for ticker, features in multi_asset_features.items():
            try:
                is_anom = self.anomaly_detector.is_anomaly(features).iloc[-1]
                anomaly_flags[ticker] = bool(is_anom)
            except Exception:
                anomaly_flags[ticker] = False

        n_anomaly = sum(anomaly_flags.values())
        logger.info(f"이상 탐지 완료: {n_anomaly}개 이상 종목")

        # ─── 6. 국면 판단 ───
        regime = "Neutral"
        regime_probs = np.array([0.33, 0.33, 0.34])
        try:
            # 대표 종목(첫 번째)의 피처로 국면 판단
            first_ticker = list(multi_asset_features.keys())[0]
            first_features = multi_asset_features[first_ticker]
            regime_probs_raw = self.regime_model.predict_proba(first_features)

            if len(regime_probs_raw) > 0:
                regime_probs = regime_probs_raw[-1]  # 최신 시점
                regime_id = int(np.argmax(regime_probs))
                regime_labels = {0: "Bull", 1: "Bear", 2: "Crash"}
                regime = regime_labels.get(regime_id, f"Regime_{regime_id}")

            logger.info(f"현재 국면: {regime} ({regime_probs})")
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")

        # ─── 7. 종합 스코어 산출 ───
        rankings = self._compute_rankings(
            multi_asset_features, fundamentals_df,
            cluster_assignments, anomaly_flags
        )

        # ─── 8. 국면 기반 종목 선정 ───
        selected = self._select_by_regime(rankings, regime, anomaly_flags)

        # 결과 생성
        result = ScreenerResult(
            timestamp=target_date or 0,
            regime=regime,
            regime_probs=regime_probs,
            selected_tickers=selected,
            cluster_assignments=cluster_assignments,
            anomaly_flags=anomaly_flags,
            rankings=rankings,
            fundamentals=fundamentals_df if not fundamentals_df.empty else pd.DataFrame(),
        )

        logger.info(f"스크리너 완료: {len(selected)}개 종목 추천")
        return result

    def _compute_rankings(self, multi_asset_features, fundamentals_df,
                           cluster_assignments, anomaly_flags) -> pd.DataFrame:
        """종합 스코어 계산 및 순위 산출."""    
        rows = []

        for ticker in multi_asset_features:
            tech_score = self._compute_tech_score(multi_asset_features[ticker])
            fund_score = 50.0  # 기본값
            tier = "C"

            if not fundamentals_df.empty and "ticker" in fundamentals_df.columns:
                fund_row = fundamentals_df[fundamentals_df["ticker"] == ticker]
                if not fund_row.empty:
                    fund_score = float(fund_row["fund_score"].iloc[0])
                    tier = str(fund_row["tier"].iloc[0])

            # 이상치 페널티
            anomaly_penalty = 20.0 if anomaly_flags.get(ticker, False) else 0.0

            # 종합 스코어
            total = (
                self.tech_weight * tech_score +
                self.fund_weight * fund_score -
                self.anomaly_penalty_weight * anomaly_penalty
            )
            total = max(0.0, min(100.0, total))

            rows.append({
                "ticker": ticker,
                "cluster": cluster_assignments.get(ticker, -1),
                "tech_score": round(tech_score, 2),
                "fund_score": round(fund_score, 2),
                "total_score": round(total, 2),
                "tier": tier,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)

        return df

    def _compute_tech_score(self, features: pd.DataFrame) -> float:
        """
        기술적 피처에서 단일 스코어 산출.
        모멘텀, 변동성, RSI를 종합하여 0~100 점수.
        """
        if features.empty:
            return 50.0

        last = features.iloc[-1]
        score = 50.0  # 중립 기저

        # 모멘텀 반영 (21일 수익률)
        ret_21d = last.get("return_21d", 0)
        if pd.notna(ret_21d):
            score += min(max(ret_21d * 200, -25), 25)  # ±25점

        # RSI 반영 (oversold/overbought)
        rsi = last.get("rsi_14", 50)
        if pd.notna(rsi):
            if rsi < 30:
                score += 10  # 과매도 → 반등 기대
            elif rsi > 70:
                score -= 10  # 과매수 → 조정 위험

        # 변동성 반영 (낮을수록 안정)
        vol = last.get("vol_21d", 0.2)
        if pd.notna(vol):
            if vol < 0.15:
                score += 5
            elif vol > 0.5:
                score -= 10

        # 허스트 지수 (추세 지속성)
        hurst = last.get("hurst_exponent", 0.5)
        if pd.notna(hurst) and hurst > 0.6:
            score += 5  # 트렌드 유리

        return max(0.0, min(100.0, score))

    def _select_by_regime(self, rankings: pd.DataFrame, regime: str,
                           anomaly_flags: Dict[str, bool]) -> List[str]:
        """국면에 따른 최종 종목 선정."""
        if rankings.empty:
            return []

        # 이상 종목 제외
        non_anomaly = rankings[
            ~rankings["ticker"].map(anomaly_flags).fillna(False)
        ]

        if non_anomaly.empty:
            non_anomaly = rankings

        # 국면별 필터 로직
        if regime == "Bull":
            # 모멘텀 + 고품질 우선
            selected = non_anomaly[non_anomaly["tier"].isin(["A", "B"])]
        elif regime == "Bear":
            # 안전 + 배당 우선
            selected = non_anomaly[non_anomaly["tier"].isin(["A", "B"])]
        elif regime == "Crash":
            # 최소 종목, 현금 비중 확대 시그널
            selected = non_anomaly.head(max(5, self.top_n // 3))
        else:
            selected = non_anomaly

        # top_n 제한
        return selected.head(self.top_n)["ticker"].tolist()

    def _extract_features_parallel(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        멀티프로세싱 기반 피처 추출.
        
        종목 간 독립적인 FeatureEngineer.extract()를 여러 CPU 코어에
        분산하여 병렬 처리. 코어 수에 비례하여 속도 향상.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import cpu_count
        
        items = list(market_data.items())
        n_tickers = len(items)
        workers = min(cpu_count() - 1, 8)  # 최대 8코어
        
        # 소수 종목일 때는 오히려 오버헤드가 크므로 직렬 처리
        if n_tickers < 50 or workers <= 1:
            return self._extract_features_serial(market_data)
        
        logger.info(f"피처 추출 시작: {n_tickers}개 종목, {workers}개 워커 프로세스")
        
        results = {}
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_ticker = {
                    executor.submit(_extract_single_ticker_features, ticker, price_df): ticker
                    for ticker, price_df in items
                }
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[ticker] = result
                    except Exception as e:
                        logger.debug(f"Feature extraction failed for {ticker}: {e}")
        except Exception as e:
            logger.warning(f"Parallel extraction failed, falling back to serial: {e}")
            return self._extract_features_serial(market_data)
        
        return results
    
    def _extract_features_serial(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """직렬 피처 추출 (폴백용)."""
        multi_asset_features = {}
        for ticker, price_df in market_data.items():
            try:
                features = self.feature_engineer.extract(price_df)
                if not features.empty and len(features) >= 20:
                    multi_asset_features[ticker] = features
            except Exception as e:
                logger.debug(f"Feature extraction failed for {ticker}: {e}")
        return multi_asset_features

    def _load_market_data(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """DB에서 시장 데이터 로드."""
        market_data = {}
        try:
            if tickers is None:
                # DB에서 가용 종목 조회
                tickers = self.db.get_available_tickers() if hasattr(self.db, 'get_available_tickers') else []

            for ticker in tickers:
                try:
                    df = self.db.load_daily_ohlcv(ticker) if hasattr(self.db, 'load_daily_ohlcv') else pd.DataFrame()
                    if not df.empty and len(df) >= 63:
                        market_data[ticker] = df
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")

        return market_data

    def _empty_result(self, timestamp: int) -> ScreenerResult:
        """빈 결과 반환."""
        return ScreenerResult(
            timestamp=timestamp,
            regime="Unknown",
            regime_probs=np.array([0.33, 0.33, 0.34]),
            selected_tickers=[],
        )

    def print_report(self, result: ScreenerResult):
        """CLI 터미널에 스크리너 리포트 출력."""
        report = result.to_cli_report()
        print(report)
