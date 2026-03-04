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

    # 국내 대표 종목 또는 거시 지표 기반 피처 추출
    fe = FeatureEngineer()
    price_df = db.query_dataframe(
        "SELECT date, open, high, low, close, volume FROM stock_daily "
        "ORDER BY date"
    )
    if price_df.empty:
        logger.error("No stock_daily data found. Run --collect first.")
        return

    features = fe.extract(price_df)
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
    ledger.create_sub_account("MidFreq", 0.15)
    ledger.create_sub_account("Swing", 0.35)
    ledger.create_sub_account("MidShort", 0.35)
    ledger.create_sub_account("Long_Safe", 0.15)

    # HMM 모델 로드 (학습 완료된 경우)
    regime_model = None
    try:
        regime_model = RegimeHMM()
        regime_model.load()
        logger.info("Regime model loaded successfully.")
    except Exception as e:
        logger.warning(f"Regime model not found, using uniform fallback: {e}")
        regime_model = None

    # 메타 포트폴리오 루프
    loop = MetaPortfolioLoop(
        ledger=ledger,
        transaction_model=TransactionModel(),
        regime_model=regime_model,
    )

    # 전략 등록
    from backend.strategies.trend_following import TrendFollowing
    from backend.strategies.long_term_value import LongTermValueStrategy

    # 시연을 위해 단순 트렌드 팔로잉과 장기 가치 전략 등록
    loop.register_strategy("Swing", TrendFollowing(fast_period=20, slow_period=50))
    loop.register_strategy("Long_Safe", LongTermValueStrategy(profile="Balanced", rebalance_freq=21))

    # 시장 데이터 로드
    tickers = db.query_dataframe(
        "SELECT DISTINCT ticker FROM stock_daily"
    )["ticker"].tolist()

    market_data = {}
    for ticker in tickers[:10]:  # 상위 10 종목으로 제한 (성능)
        df = db.query_dataframe(
            f"SELECT date, open, high, low, close, volume "
            f"FROM stock_daily WHERE ticker='{ticker}' ORDER BY date"
        )
        if not df.empty:
            market_data[ticker] = df

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
            for _, row in curve_df.iterrows():
                ts = str(int(row['timestamp']))
                date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:]}" if len(ts) == 8 else ts
                eq_data.append({"date": date_str, "equity": float(row['equity'])})
        
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
            "equityCurve": eq_data[-100:] if len(eq_data) > 100 else eq_data
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


def main():
    parser = argparse.ArgumentParser(description="AutoML Quant Trade Pipeline")
    parser.add_argument("--collect-insert", action="store_true", help="Run data collection for new tickers only (Insert)")
    parser.add_argument("--collect-update", action="store_true", help="Run data collection for existing tickers only (Update)")
    parser.add_argument("--collect-macro", action="store_true", help="Collect macro indicators only")
    parser.add_argument("--collect-overseas", action="store_true", help="Collect overseas assets only")
    parser.add_argument("--train-regime", action="store_true", help="Train regime detection model (Phase 2)")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine (Phase 3)")
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

    if args.server:
        logger.info("Starting Dashboard Backend Server (FastAPI)")
        import uvicorn
        uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=True)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
