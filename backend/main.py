"""
AutoML_Quant_Trade - 파이프라인 오케스트레이터

CLI 인터페이스를 통해 데이터 수집, 국면 감지, 백테스팅 등
전체 파이프라인을 순차적으로 실행.

Usage:
    python -m backend.main --collect-insert    # DB에 없는 종목 전체 데이터 수집 (최초)
    python -m backend.main --collect-update    # DB에 있는 종목 최근 데이터 수집 (업데이트)
    python -m backend.main --collect-macro     # 거시지표만 수집
    python -m backend.main --collect-overseas   # 해외 자산만 수집
    python -m backend.main --train-regime      # 국면 모델 학습 (Phase 2)
    python -m backend.main --backtest          # 백테스팅 실행 (Phase 3)
    python -m backend.main --screen            # 스크리너 단독 실행
    python -m backend.main --server            # 대시보드 백엔드 구동 (FastAPI)
"""
import argparse
import logging
import sys

from backend.config.settings import Settings
from backend.data.database import DatabaseManager
from backend.data.bridge_client import BridgeClient
from backend.data.stock_collector import StockCollector
from backend.data.overseas_collector import OverseasCollector
from backend.data.macro_collector import MacroCollector
from backend.data.asset_universe import AssetUniverseMapper
from backend.utils.logger import setup_integrated_logger

# DB 객체를 메인에서 먼저 생성하여 로거 설정에 주입
# (만약 초기화 시점이 맞지 않으면 전역 세팅 방식 고민이 필요하지만, 여기서는 main 진입 시점 생성)
db = DatabaseManager()
logger = setup_integrated_logger(db, source="backend")



def run_collect_insert(db: DatabaseManager, client: BridgeClient):
    """신규 종목 데이터 수집 (Insert): 국내 주식 + 해외 자산 + 거시지표"""
    logger.info("=" * 60)
    logger.info("Starting data collection pipeline (INSERT)")
    logger.info("=" * 60)

    # 1. 국내 주식 유니버스 수집
    logger.info("[1/5] Fetching domestic equity universe...")
    try:
        universe = client.fetch_universe()
        logger.info(f"  → {len(universe)} domestic tickers found")
    except Exception as e:
        logger.error(f"  → Failed to fetch universe: {e}")
        universe = []

    # 2. 해외 유니버스 동적 수집 (CpUsCode)
    logger.info("[2/5] Fetching overseas universe from CpUsCode...")
    overseas_codes = []
    try:
        # 국가대표 지수 (UsType=2), 해외 개별주식 (4), 원자재 (6), 환율 (7)
        for us_type, label in [(2, "국가대표지수"), (4, "해외개별주식"),
                                (6, "원자재/반도체"), (7, "환율")]:
            codes = client.fetch_overseas_universe(us_type)
            logger.info(f"  → {label} (UsType={us_type}): {len(codes)}개")
            overseas_codes.extend(codes)

        # AssetUniverseMapper의 정적 코드도 병합 (중복 제거)
        mapper = AssetUniverseMapper()
        static_codes = mapper.get_codes_by_source("overseas")
        overseas_codes = list(dict.fromkeys(overseas_codes + static_codes))  # 순서 유지 중복 제거
        logger.info(f"  → 총 해외 유니버스: {len(overseas_codes)}개")
    except Exception as e:
        logger.error(f"  → Failed to fetch overseas universe: {e}")
        mapper = AssetUniverseMapper()
        overseas_codes = mapper.get_codes_by_source("overseas")
        logger.info(f"  → Fallback to static codes: {len(overseas_codes)}개")

    # 3. 국내 주식 일봉 수집 (신규)
    if universe:
        logger.info("[3/5] Collecting domestic daily OHLCV (Insert)...")
        collector = StockCollector(db=db, client=client)
        collector.collect_daily_insert(universe)

    # 4. 해외 자산 일봉 수집 (신규)
    if overseas_codes:
        logger.info("[4/5] Collecting overseas assets (Insert)...")
        overseas_collector = OverseasCollector(db=db, client=client)
        overseas_collector.collect_insert(overseas_codes)

    # 5. 거시지표 수집 (신규)
    logger.info("[5/5] Collecting macro indicators (Insert)...")
    macro_collector = MacroCollector(db=db, client=client)
    macro_collector.collect_insert()

    # 요약
    logger.info("=" * 60)
    logger.info("Insert Collection complete. Database summary:")
    logger.info(f"  stock_daily:    {db.get_row_count('stock_daily'):>10,} rows")
    logger.info(f"  stock_minute:   {db.get_row_count('stock_minute'):>10,} rows")
    logger.info(f"  overseas_daily: {db.get_row_count('overseas_daily'):>10,} rows")
    logger.info(f"  macro_daily:    {db.get_row_count('macro_daily'):>10,} rows")
    logger.info("=" * 60)


def run_collect_update(db: DatabaseManager, client: BridgeClient):
    """기존 종목 데이터 업데이트 (Update): 국내 주식 + 해외 자산 + 거시지표"""
    logger.info("=" * 60)
    logger.info("Starting data collection pipeline (UPDATE)")
    logger.info("=" * 60)

    # 1. 국내 주식 유니버스 수집
    logger.info("[1/5] Fetching domestic equity universe...")
    try:
        universe = client.fetch_universe()
        logger.info(f"  → {len(universe)} domestic tickers found")
    except Exception as e:
        logger.error(f"  → Failed to fetch universe: {e}")
        universe = []

    # 2. 해외 유니버스 동적 수집 (CpUsCode)
    logger.info("[2/5] Fetching overseas universe from CpUsCode...")
    overseas_codes = []
    try:
        for us_type, label in [(2, "국가대표지수"), (4, "해외개별주식"),
                                (6, "원자재/반도체"), (7, "환율")]:
            codes = client.fetch_overseas_universe(us_type)
            logger.info(f"  → {label} (UsType={us_type}): {len(codes)}개")
            overseas_codes.extend(codes)

        mapper = AssetUniverseMapper()
        static_codes = mapper.get_codes_by_source("overseas")
        overseas_codes = list(dict.fromkeys(overseas_codes + static_codes))
        logger.info(f"  → 총 해외 유니버스: {len(overseas_codes)}개")
    except Exception as e:
        logger.error(f"  → Failed to fetch overseas universe: {e}")
        mapper = AssetUniverseMapper()
        overseas_codes = mapper.get_codes_by_source("overseas")
        logger.info(f"  → Fallback to static codes: {len(overseas_codes)}개")

    # 3. 국내 주식 일봉 수집 (업데이트)
    if universe:
        logger.info("[3/5] Collecting domestic daily OHLCV (Update)...")
        collector = StockCollector(db=db, client=client)
        collector.collect_daily_update(universe)

    # 4. 해외 자산 일봉 수집 (업데이트)
    if overseas_codes:
        logger.info("[4/5] Collecting overseas assets (Update)...")
        overseas_collector = OverseasCollector(db=db, client=client)
        overseas_collector.collect_update(overseas_codes)

    # 5. 거시지표 수집 (업데이트)
    logger.info("[5/5] Collecting macro indicators (Update)...")
    macro_collector = MacroCollector(db=db, client=client)
    macro_collector.collect_update()

    # 요약
    logger.info("=" * 60)
    logger.info("Update Collection complete. Database summary:")
    logger.info(f"  stock_daily:    {db.get_row_count('stock_daily'):>10,} rows")
    logger.info(f"  stock_minute:   {db.get_row_count('stock_minute'):>10,} rows")
    logger.info(f"  overseas_daily: {db.get_row_count('overseas_daily'):>10,} rows")
    logger.info(f"  macro_daily:    {db.get_row_count('macro_daily'):>10,} rows")
    logger.info("=" * 60)


def run_collect_macro(db: DatabaseManager, client: BridgeClient):
    """거시지표만 수집"""
    logger.info("Collecting macro indicators...")
    macro_collector = MacroCollector(db=db, client=client)
    macro_collector.collect_all()


def run_collect_overseas(db: DatabaseManager, client: BridgeClient):
    """해외 자산만 수집"""
    logger.info("Collecting overseas assets...")

    overseas_codes = []
    try:
        for us_type, label in [(2, "국가대표지수"), (4, "해외개별주식"),
                                (6, "원자재/반도체"), (7, "환율")]:
            codes = client.fetch_overseas_universe(us_type)
            logger.info(f"  → {label} (UsType={us_type}): {len(codes)}개")
            overseas_codes.extend(codes)

        mapper = AssetUniverseMapper()
        static_codes = mapper.get_codes_by_source("overseas")
        overseas_codes = list(dict.fromkeys(overseas_codes + static_codes))
        logger.info(f"  → 총 해외 유니버스: {len(overseas_codes)}개")
    except Exception as e:
        logger.error(f"  → Failed to fetch overseas universe: {e}")
        mapper = AssetUniverseMapper()
        overseas_codes = mapper.get_codes_by_source("overseas")

    overseas_collector = OverseasCollector(db=db, client=client)
    overseas_collector.collect_batch(overseas_codes)


def run_train_regime(db: DatabaseManager):
    """국면 모델 학습 (Phase 2)"""
    logger.info("Phase 2: Regime detection training")
    from backend.models.feature_engineer import FeatureEngineer
    from backend.models.regime_hmm import RegimeHMM
    from backend.data.macro_collector import MacroCollector

    # 국내 대표 종목(A069500: KOSPI 200) 및 거시 지표 기반 피처 추출
    fe = FeatureEngineer()
    price_df = db.query_dataframe(
        "SELECT date, open, high, low, close, volume FROM stock_daily "
        "WHERE ticker = 'A069500' ORDER BY date"
    )
    if price_df.empty:
        logger.error("No stock_daily data found for benchmark (A069500).")
        return

    macro_df = db.query_dataframe("SELECT * FROM macro_daily ORDER BY date")
    macro_wide = macro_df.pivot_table(index='date', columns='indicator', values='close').reset_index() if not macro_df.empty else None
    
    features = fe.extract(price_df, macro_wide)
    if features.empty:
        logger.error("Feature extraction produced empty results.")
        return

    hmm = RegimeHMM()
    hmm.fit(features)
    hmm.save()
    logger.info(f"Regime model saved. Transition matrix:\n{hmm.get_transition_matrix()}")


def run_backtest(db: DatabaseManager):
    """백테스팅 실행 (Phase 3)"""
    logger.info("Phase 3+4: Backtesting with Meta Portfolio Engine")
    from backend.engine.ledger import MasterLedger
    from backend.engine.transaction_model import TransactionModel
    from backend.meta.meta_portfolio_loop import MetaPortfolioLoop
    from backend.models.regime_hmm import RegimeHMM

    # 원장 세팅
    ledger = MasterLedger(initial_capital=Settings.INITIAL_CAPITAL)
    ledger.create_sub_account("Regime", 0.30)
    ledger.create_sub_account("Anomaly", 0.25)
    ledger.create_sub_account("Cluster", 0.25)
    ledger.create_sub_account("Long_Safe", 0.20)

    # HMM 모델 로드 (학습 완료된 경우)
    regime_model = None
    try:
        regime_model = RegimeHMM()
        regime_model.load()
        logger.info("Regime model loaded successfully.")
    except Exception as e:
        logger.warning(f"Regime model not found, using uniform fallback: {e}")
        regime_model = None

    # 시장 데이터 선행 로드 (국내 주식 + 해외 유니버스 모두)
    from backend.data.asset_universe import AssetUniverseMapper
    mapper = AssetUniverseMapper()
    target_codes_overseas = mapper.get_codes_by_source("overseas")
    target_codes_domestic_etf = mapper.get_codes_by_source("domestic")
    
    # KOSPI 일반 주식 중 최근 거래량이 어느정도 있는 종목만 필터링 (슬리피지 파산 방지)
    all_domestic_tickers = db.query_dataframe(
        "SELECT ticker FROM stock_daily GROUP BY ticker HAVING max(volume) >= 50000"
    )["ticker"].tolist()
    
    all_domestic_targets = list(set(all_domestic_tickers + target_codes_domestic_etf))
    
    market_data = {}
    
    logger.info(f"Loading {len(all_domestic_targets)} domestic tickers into memory...")
    for ticker in all_domestic_targets:
        df = db.query_dataframe(
            f"SELECT date, open, high, low, close, volume "
            f"FROM stock_daily WHERE ticker='{ticker}' ORDER BY date"
        )
        if not df.empty:
            market_data[ticker] = df

    all_overseas_codes = db.query_dataframe(
        "SELECT code FROM overseas_daily GROUP BY code HAVING max(volume) >= 10000"
    )["code"].tolist()
    
    all_overseas_targets = list(set(all_overseas_codes + target_codes_overseas))
    
    logger.info(f"Loading {len(all_overseas_targets)} overseas codes into memory...")
    for code in all_overseas_targets:
        df = db.query_dataframe(
            f"SELECT date, open, high, low, close, volume "
            f"FROM overseas_daily WHERE code='{code}' ORDER BY date"
        )
        if not df.empty:
            market_data[code] = df

    if not market_data:
        logger.error("No market data available. Run --collect first.")
        return

    # 거시지표 로드 및 Wide Form 변환 (HMM 피처용)
    macro_data = db.query_dataframe(
        "SELECT * FROM macro_daily ORDER BY date"
    )
    macro_wide = macro_data.pivot_table(index='date', columns='indicator', values='close').reset_index() if not macro_data.empty else None

    # 🚀 성능 최적화: KOSPI 200 (A069500) 기준으로 국면을 루프 밖에서 1회 사전 계산
    logger.info("Pre-computing HMM Regimes to optimize backtest speed...")
    from backend.models.feature_engineer import FeatureEngineer
    import numpy as np
    
    benchmark_ticker = "A069500" if "A069500" in market_data else (list(market_data.keys())[0] if market_data else None)
    precomputed_regimes = {}

    if benchmark_ticker and regime_model is not None:
        fe = FeatureEngineer()
        benchmark_df = market_data[benchmark_ticker]
        
        features_df = fe.extract(benchmark_df, macro_wide)
        if not features_df.empty:
            # 빠른 전 구간 HMM 예측 (Walk-Forward)
            regime_pred_df = regime_model.walk_forward_predict(features_df, train_window=252, retrain_freq=21)
            
            # { timestamp: np.array([prob_0, prob_1, prob_2]) } 형태로 매핑
            for _, row in regime_pred_df.iterrows():
                ts = int(row["date"])
                probs = np.array([row[f"prob_{i}"] for i in range(Settings.REGIME_COUNT)])
                precomputed_regimes[ts] = probs
            
            logger.info(f"Pre-computed regimes for {len(precomputed_regimes)} dates using {benchmark_ticker}.")
        else:
            logger.warning("Feature extraction failed for benchmark. Regimes will fallback to uniform.")

    # 메타 포트폴리오 루프
    loop = MetaPortfolioLoop(
        ledger=ledger,
        transaction_model=TransactionModel(),
        regime_model=None, # 더 이상 루프 내에서 HMM 모델을 사용하지 않음
        precomputed_regimes=precomputed_regimes # 새로 추가된 인자
    )

    # 전략 등록 (비지도학습 기반)
    from backend.strategies.regime_adaptive import RegimeAdaptiveStrategy
    from backend.strategies.anomaly_strategy import AnomalyStrategy
    from backend.strategies.cluster_momentum import ClusterMomentumStrategy
    from backend.strategies.long_term_value import LongTermValueStrategy
    from backend.models.anomaly_detector import AnomalyDetector
    from backend.models.cluster_analyzer import CrossAssetClusterAnalyzer

    # Anomaly Detector & Cluster Analyzer 학습
    anomaly_model = AnomalyDetector()
    cluster_analyzer = CrossAssetClusterAnalyzer()

    try:
        anomaly_model.load()
        logger.info("AnomalyDetector loaded.")
    except Exception:
        logger.info("AnomalyDetector not found. Training...")
        benchmark_features = None
        if benchmark_ticker:
            fe = FeatureEngineer()
            benchmark_features = fe.extract(market_data[benchmark_ticker], macro_wide)
        if benchmark_features is not None and not benchmark_features.empty:
            anomaly_model.fit(benchmark_features)
            anomaly_model.save()

    try:
        cluster_analyzer.load()
        logger.info("ClusterAnalyzer loaded.")
    except Exception:
        logger.info("ClusterAnalyzer not found. Training...")
        fe = FeatureEngineer()
        multi_features = {}
        for t, pdf in list(market_data.items())[:200]:  # 상위 200개 종목으로 제한
            try:
                ft = fe.extract(pdf)
                if not ft.empty and len(ft) >= 20:
                    multi_features[t] = ft
            except Exception:
                continue
        if len(multi_features) >= 10:
            cluster_analyzer.fit(multi_features)
            cluster_analyzer.save()

    # 스크리너 구성 (백테스트 中 리밸런싱 주기마다 종목 풀 갱신)
    from backend.screener.fundamental_scorer import FundamentalScorer
    from backend.screener.screener import UnsupervisedScreener

    screener = UnsupervisedScreener(
        db=db,
        cluster_analyzer=cluster_analyzer,
        anomaly_detector=anomaly_model,
        regime_model=regime_model or RegimeHMM(),
        fundamental_scorer=FundamentalScorer(),  # BridgeClient 없으면 빈 재무 데이터
    )

    # 전략 등록
    loop.register_strategy("Regime", RegimeAdaptiveStrategy(
        regime_model=regime_model or RegimeHMM(),
    ))
    loop.register_strategy("Anomaly", AnomalyStrategy(
        anomaly_detector=anomaly_model,
    ))
    loop.register_strategy("Cluster", ClusterMomentumStrategy(
        cluster_analyzer=cluster_analyzer,
    ))
    loop.register_strategy("Long_Safe", LongTermValueStrategy(
        profile="Balanced", rebalance_freq=21,
    ))

    # 스크리너 연동: 21 거래일마다 종목 풀 재계산 후 전략에 주입
    screener_market_data = {
        t: df for t, df in market_data.items()
        if t.startswith("A")  # 국내 주식만 스크리너 대상
    }
    loop.set_screener(
        screener=screener,
        market_data=screener_market_data,
        refresh_freq=21,
    )

    # 시장 데이터 로드 (국내 주식 + 해외 유니버스 모두)
    from backend.data.asset_universe import AssetUniverseMapper
    mapper = AssetUniverseMapper()
    target_codes_overseas = mapper.get_codes_by_source("overseas")
    target_codes_domestic_etf = mapper.get_codes_by_source("domestic")
    
    # KOSPI 일반 주식 중 최근 거래량이 어느정도 있는 종목만 필터링 (슬리피지 파산 방지)
    # 서브쿼리를 사용하여 최대 거래량이 10000주 이상인 종목들만 대상으로 함
    all_domestic_tickers = db.query_dataframe(
        "SELECT ticker FROM stock_daily GROUP BY ticker HAVING max(volume) >= 50000"
    )["ticker"].tolist()
    
    # 중복 제거 (ETF 포함)
    all_domestic_targets = list(set(all_domestic_tickers + target_codes_domestic_etf))
    
    market_data = {}
    
    # === 국내 데이터 로드 ===
    logger.info(f"Loading {len(all_domestic_targets)} domestic tickers into memory...")
    for ticker in all_domestic_targets:
        df = db.query_dataframe(
            f"SELECT date, open, high, low, close, volume "
            f"FROM stock_daily WHERE ticker='{ticker}' ORDER BY date"
        )
        if not df.empty:
            market_data[ticker] = df

    # === 해외 자산 전체 데이터 로드 ===
    # 해외 자산도 거래량이 어느 정도 있는 종목만 포섭
    all_overseas_codes = db.query_dataframe(
        "SELECT code FROM overseas_daily GROUP BY code HAVING max(volume) >= 10000"
    )["code"].tolist()
    
    all_overseas_targets = list(set(all_overseas_codes + target_codes_overseas))
    
    logger.info(f"Loading {len(all_overseas_targets)} overseas codes into memory...")
    for code in all_overseas_targets:
        df = db.query_dataframe(
            f"SELECT date, open, high, low, close, volume "
            f"FROM overseas_daily WHERE code='{code}' ORDER BY date"
        )
        if not df.empty:
            market_data[code] = df

    if not market_data:
        logger.error("No market data available. Run --collect first.")
        return

    equity_curve = loop.run(market_data)
    logger.info(f"Equity curve: {len(equity_curve)} data points")

    # --- [NEW] 백테스트 결과 직렬화 로직 ---
    import json
    import numpy as np
    from pathlib import Path
    
    cache_dir = Path("cache_daishin")
    cache_dir.mkdir(exist_ok=True)
    
    algorithms = []
    for idx, (name, acc) in enumerate(ledger.sub_accounts.items(), 1):
        metrics = acc.get_performance_metrics()
        curve_df = acc.get_equity_curve()
        
        eq_data = []
        if not curve_df.empty:
            # 병목 해소: iterrows() 대신 vectorization 사용 및 마지막 100개만 변환
            tail_df = curve_df.tail(100)
            timestamps = tail_df['timestamp'].astype(int).astype(str).values
            equities = tail_df['equity'].astype(float).values
            
            for t_str, eq_val in zip(timestamps, equities):
                date_str = f"{t_str[:4]}-{t_str[4:6]}-{t_str[6:]}" if len(t_str) == 8 else t_str
                eq_data.append({"date": date_str, "equity": eq_val})
        
        algorithms.append({
            "id": str(idx),
            "name": name,
            "timeframe": "Daily",
            "rank": idx,
            "metrics": {
                "cumulativeReturn": float(metrics.get("total_return", 0) * 100),
                "maxDrawdown": float(metrics.get("max_drawdown", 0) * 100),
                "sharpeRatio": float(metrics.get("sharpe_ratio", 0)),
                "winRate": float(metrics.get("win_rate", 0) * 100)
            },
            "equityCurve": eq_data
        })
        
    master_metrics = ledger.get_performance_metrics()
    probs = loop.last_regime_probs if loop.last_regime_probs is not None else [0.33, 0.33, 0.34]
    regime_labels = ["Bull", "Bear", "Crash"]
    dominant_idx = int(np.argmax(probs))
    
    last_date = "1970-01-01"
    if not equity_curve.empty:
        ts = str(int(equity_curve.iloc[-1]["timestamp"]))
        last_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:]}" if len(ts) == 8 else ts
        
    dashboard_data = {
        "currentRegime": {
            "timestamp": last_date,
            "probabilities": {
                "Bull": float(probs[0]),
                "Bear": float(probs[1]),
                "Crash": float(probs[2])
            },
            "dominantRegime": regime_labels[dominant_idx]
        },
        "masterMetrics": {
            "initialCapital": float(master_metrics.get("initial_capital", 0)),
            "totalReturn": float(master_metrics.get("total_return", 0) * 100),
        },
        "algorithms": algorithms
    }
    
    out_path = cache_dir / "latest_backtest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Dashboard data saved to {out_path}")


def run_screen(db: DatabaseManager):
    """스크리너 단독 실행 — CLI 리포트 + JSON 캐시 출력"""
    import json
    from pathlib import Path
    import numpy as np
    from backend.models.feature_engineer import FeatureEngineer
    from backend.models.regime_hmm import RegimeHMM
    from backend.models.anomaly_detector import AnomalyDetector
    from backend.models.cluster_analyzer import CrossAssetClusterAnalyzer
    from backend.screener.fundamental_scorer import FundamentalScorer
    from backend.screener.screener import UnsupervisedScreener

    logger.info("=" * 60)
    logger.info("Screener Execution")
    logger.info("=" * 60)

    # 모델 로드
    regime_model = RegimeHMM()
    try:
        regime_model.load()
    except Exception as e:
        logger.warning(f"Regime model not found: {e}")

    anomaly_model = AnomalyDetector()
    try:
        anomaly_model.load()
    except Exception as e:
        logger.warning(f"AnomalyDetector not found: {e}")

    cluster_analyzer = CrossAssetClusterAnalyzer()
    try:
        cluster_analyzer.load()
    except Exception as e:
        logger.warning(f"ClusterAnalyzer not found: {e}")

    # 시장 데이터 로드
    tickers = db.query_dataframe(
        "SELECT ticker FROM stock_daily GROUP BY ticker HAVING max(volume) >= 50000"
    )["ticker"].tolist()

    market_data = {}
    for ticker in tickers[:300]:  # 상위 300개 제한
        df = db.query_dataframe(
            f"SELECT date, open, high, low, close, volume "
            f"FROM stock_daily WHERE ticker='{ticker}' ORDER BY date"
        )
        if not df.empty and len(df) >= 63:
            market_data[ticker] = df

    # FundamentalScorer (BridgeClient 없이 실행 시 빈 결과)
    scorer = FundamentalScorer()

    # 스크리너 실행
    screener = UnsupervisedScreener(
        db=db,
        cluster_analyzer=cluster_analyzer,
        anomaly_detector=anomaly_model,
        regime_model=regime_model,
        fundamental_scorer=scorer,
    )

    result = screener.run(market_data=market_data, tickers=tickers[:300])

    # CLI 리포트 출력
    screener.print_report(result)

    # JSON 캐시 저장 (프론트엔드 대시보드용)
    cache_dir = Path("cache_daishin")
    cache_dir.mkdir(exist_ok=True)
    result.save_json(str(cache_dir / "latest_screener.json"))
    logger.info(f"Screener results saved to {cache_dir / 'latest_screener.json'}")


def main():
    parser = argparse.ArgumentParser(description="AutoML Quant Trade Pipeline")
    parser.add_argument("--collect-insert", action="store_true", help="Run data collection for new tickers only (Insert)")
    parser.add_argument("--collect-update", action="store_true", help="Run data collection for existing tickers only (Update)")
    parser.add_argument("--collect-macro", action="store_true", help="Collect macro indicators only")
    parser.add_argument("--collect-overseas", action="store_true", help="Collect overseas assets only")
    parser.add_argument("--train-regime", action="store_true", help="Train regime detection model (Phase 2)")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine (Phase 3)")
    parser.add_argument("--screen", action="store_true", help="Run unsupervised screener")
    parser.add_argument("--server", action="store_true", help="Start FastAPI Backend Server for Dashboard")
    parser.add_argument("--db-info", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    args = parser.parse_args()

    # 디렉토리 생성
    Settings.ensure_dirs()

    if args.db_info:
        print(f"Database: {Settings.DB_PATH}")
        print(f"  stock_daily:    {db.get_row_count('stock_daily'):>10,} rows")
        print(f"  stock_minute:   {db.get_row_count('stock_minute'):>10,} rows")
        print(f"  overseas_daily: {db.get_row_count('overseas_daily'):>10,} rows")
        print(f"  macro_daily:    {db.get_row_count('macro_daily'):>10,} rows")
        return


    # 브릿지 필요한 작업들
    if args.collect_insert or args.collect_update or args.collect_macro or args.collect_overseas:
        client = BridgeClient()
        try:
            if args.collect_insert:
                run_collect_insert(db, client)
            if args.collect_update:
                run_collect_update(db, client)
            elif args.collect_macro:
                run_collect_macro(db, client)
            elif args.collect_overseas:
                run_collect_overseas(db, client)
        finally:
            client.close()
        return

    if args.train_regime:
        run_train_regime(db)
        return

    if args.backtest:
        run_backtest(db)
        return

    if args.screen:
        run_screen(db)
        return

    if args.server:
        logger.info("Starting Dashboard Backend Server (FastAPI)")
        import uvicorn
        uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=True)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
