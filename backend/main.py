"""
AutoML_Quant_Trade - 파이프라인 오케스트레이터

CLI 인터페이스를 통해 데이터 수집, 국면 감지, 백테스팅 등
전체 파이프라인을 순차적으로 실행.

Usage:
    python -m backend.main --collect           # 전체 데이터 수집
    python -m backend.main --collect-macro     # 거시지표만 수집
    python -m backend.main --collect-overseas   # 해외 자산만 수집
    python -m backend.main --train-regime      # 국면 모델 학습 (Phase 2)
    python -m backend.main --backtest          # 백테스팅 실행 (Phase 3)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def run_collect_all(db: DatabaseManager, client: BridgeClient):
    """전체 데이터 수집: 국내 주식 + 해외 자산 + 거시지표"""
    logger.info("=" * 60)
    logger.info("Starting full data collection pipeline")
    logger.info("=" * 60)

    # 1. 국내 주식 유니버스 수집
    logger.info("[1/4] Fetching domestic equity universe...")
    try:
        universe = client.fetch_universe()
        logger.info(f"  → {len(universe)} domestic tickers found")
    except Exception as e:
        logger.error(f"  → Failed to fetch universe: {e}")
        universe = []

    # 2. 국내 주식 일봉 수집
    if universe:
        logger.info("[2/4] Collecting domestic daily OHLCV...")
        collector = StockCollector(db=db, client=client)
        collector.collect_daily_incremental(universe)

    # 3. 해외 자산 일봉 수집
    logger.info("[3/4] Collecting overseas assets...")
    mapper = AssetUniverseMapper()
    overseas_codes = mapper.get_codes_by_source("overseas")
    if overseas_codes:
        overseas_collector = OverseasCollector(db=db, client=client)
        overseas_collector.collect_batch(overseas_codes)

    # 4. 거시지표 수집
    logger.info("[4/4] Collecting macro indicators...")
    macro_collector = MacroCollector(db=db, client=client)
    macro_collector.collect_all()

    # 요약
    logger.info("=" * 60)
    logger.info("Collection complete. Database summary:")
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
    mapper = AssetUniverseMapper()
    overseas_codes = mapper.get_codes_by_source("overseas")

    overseas_collector = OverseasCollector(db=db, client=client)
    overseas_collector.collect_batch(overseas_codes)


def main():
    parser = argparse.ArgumentParser(description="AutoML Quant Trade Pipeline")
    parser.add_argument("--collect", action="store_true", help="Run full data collection")
    parser.add_argument("--collect-macro", action="store_true", help="Collect macro indicators only")
    parser.add_argument("--collect-overseas", action="store_true", help="Collect overseas assets only")
    parser.add_argument("--train-regime", action="store_true", help="Train regime detection model (Phase 2)")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine (Phase 3)")
    parser.add_argument("--db-info", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    # 디렉토리 생성
    Settings.ensure_dirs()

    # DB 초기화
    db = DatabaseManager()

    if args.db_info:
        print(f"Database: {Settings.DB_PATH}")
        print(f"  stock_daily:    {db.get_row_count('stock_daily'):>10,} rows")
        print(f"  stock_minute:   {db.get_row_count('stock_minute'):>10,} rows")
        print(f"  overseas_daily: {db.get_row_count('overseas_daily'):>10,} rows")
        print(f"  macro_daily:    {db.get_row_count('macro_daily'):>10,} rows")
        return

    # 브릿지 필요한 작업들
    if args.collect or args.collect_macro or args.collect_overseas:
        client = BridgeClient()
        try:
            if args.collect:
                run_collect_all(db, client)
            elif args.collect_macro:
                run_collect_macro(db, client)
            elif args.collect_overseas:
                run_collect_overseas(db, client)
        finally:
            client.close()
        return

    if args.train_regime:
        logger.info("Phase 2: Regime detection training — not yet implemented")
        return

    if args.backtest:
        logger.info("Phase 3: Backtesting engine — not yet implemented")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
