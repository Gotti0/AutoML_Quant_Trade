"""
AutoML_Quant_Trade - M2 모델링 통합 백테스터 실행 스크립트

FeatureEngineer 추출값 검증 및 Stable Baselines3(PPO) 기반 
장 초반 단기 매매 학습/백테스팅 파이프라인.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from backend.config.settings import Settings
from backend.data.database import DatabaseManager
from backend.models.feature_engineer import FeatureEngineer
from backend.models.rl_agent import IntradayTradingEnv, RLBaselineModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(ticker: str = "A005930", train_steps: int = 5000):
    """
    1. 데이터베이스에서 주식 데이터 로드
    2. FeatureEngineer를 통한 학습 피처 + 단기 수익률 병합
    3. 강화학습 환경 구축 및 PPO 에이전트 학습
    4. 학습된 모델로 시뮬레이션 및 결과 시각화
    """
    db = DatabaseManager()
    
    # 데이터 로드
    logger.info(f"Loading data for {ticker}...")
    daily_df = db.load_stock_daily(ticker)
    minute_df = db.load_stock_minute(ticker)
    macro_df = db.load_macro_all()

    if daily_df.empty or minute_df.empty:
        logger.error(f"Insufficient data for {ticker}. Please run data collection first.")
        return

    # 피처 추출
    logger.info("Extracting features (FeatureEngineer)...")
    fe = FeatureEngineer()
    featured_df = fe.extract(price_df=daily_df, minute_df=minute_df, macro_df=macro_df)
    
    # 단기 타겟값이 없으면 훈련 불가 (장 초반 09:00~09:30 데이터 결측치)
    if "intraday_return_30m" not in featured_df.columns:
        logger.error("'intraday_return_30m' target feature not found.")
        return
        
    featured_df = featured_df.dropna().reset_index(drop=True)
    logger.info(f"Final training dataset size: {len(featured_df)} rows")

    if len(featured_df) < 50:
        logger.error(f"Not enough valid rows after dropping NaNs ({len(featured_df)}).")
        return

    # Train / Test 분할 (단순 시계열 8:2)
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx].reset_index(drop=True)
    test_df = featured_df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Dataset split -> Train: {len(train_df)}, Test: {len(test_df)}")

    # 훈련 환경 구축 및 학습
    train_env = IntradayTradingEnv(df=train_df)
    rl_agent = RLBaselineModel(train_env)
    
    logger.info(f"Training PPO Agent for {train_steps} steps...")
    rl_agent.train(total_timesteps=train_steps)
    
    # 모델 저장
    model_path = os.path.join(Settings.MODEL_DIR, f"ppo_baseline_{ticker}")
    rl_agent.save(model_path)
    
    # 테스트 환경에서 평가 (백테스트)
    logger.info("Evaluating on Test Dataset...")
    test_env = IntradayTradingEnv(df=test_df)
    
    # 테스트 환경용으로 에이전트에 환경 교체 후 실행
    # (Stable Baselines3 특성상 새로운 환경을 래핑)
    rl_agent.model.set_env(test_env)
    eval_result = rl_agent.evaluate()
    
    logger.info(f"Final Test Balance: {eval_result['balance'].iloc[-1]:,.0f}")
    
    # 수익률 커브 저장
    save_plot(eval_result, ticker)


def save_plot(df: pd.DataFrame, ticker: str):
    """테스트 평가 결과를 시각화하여 저장"""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'].astype(str), df['balance'], label='Portfolio Balance')
        plt.title(f'RL Agent Backtest Result - {ticker} (Out-of-Sample)')
        plt.xlabel('Date')
        plt.ylabel('Balance (KRW)')
        
        # X축 라벨 너무 많으면 안 보이므로 적당히 스킵
        n_ticks = max(1, len(df) // 10)
        plt.xticks(df['date'].astype(str)[::n_ticks], rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(Settings.PROJECT_ROOT, "docs", f"rl_backtest_{ticker}.png")
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Plot saving failed: {e}")

if __name__ == "__main__":
    # 실행 테스트: 삼성전자 기준 2000 스텝 훈련
    run_pipeline(ticker="A005930", train_steps=2000)
    
    logger.info("\n=== M2 Phase 1차 통합 파이프라인(베타) 실행 완료 ===")
