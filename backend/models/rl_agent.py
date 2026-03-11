"""
AutoML_Quant_Trade - 강화학습(Reinforcement Learning) 베이스라인 모델

장 초반(09:00~09:30) 단기 매매에 특화된 OpenAI Gym 호환 커스텀 환경과 
Stable Baselines3 PPO 에이전트를 통한 학습 파이프라인.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class IntradayTradingEnv(gym.Env):
    """
    장 초반 09:00 시초가에 진입하여 09:30 종가에 청산하는
    단기 퀀트 트레이딩을 위한 강화학습 환경.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 1_000_000.0):
        """
        초기화
        Parameters:
            df: FeatureEngineer를 통해 추출된 특성 벡트와 intraday_return_30m이 포함된 DataFrame
        """
        super(IntradayTradingEnv, self).__init__()
        
        self.df = df.dropna().reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # 제외할 컬럼(시계열 정보, 6개의 청산 타겟 수익률 등) 식별 및 Feature 추출
        self.target_cols = [f"ret_{t}" for t in [905, 910, 915, 920, 925, 930]]
        self.exclude_cols = ["date", "ticker"] + self.target_cols + ["intraday_return_30m"]
        
        # 필수 타겟 Feature 필터링
        for t_col in self.target_cols:
            if t_col not in self.df.columns:
                logger.warning(f"Target column missing: {t_col}. Padding with 0.")
                self.df[t_col] = 0.0

        self.feature_cols = [c for c in self.df.columns if c not in self.exclude_cols]
        
        if len(self.df) == 0:
            raise ValueError("Valid data is empty after dropping NaNs.")

        # --- 행동 공간 (Action Space) ---
        # 0: 관망
        # 1: 매수 후 09:05 청산
        # 2: 매수 후 09:10 청산
        # ...
        # 6: 매수 후 09:30 청산
        self.action_space = spaces.Discrete(7)

        # --- 상태 공간 (Observation Space) ---
        # n개의 Feature Vector
        self.obs_dim = len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # 에피소드 상태 변수
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.balance = self.initial_balance
        self.history: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """한 에피소드(시계열 처음) 시작 (Gymnasium v26+ API)"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.history: List[Dict[str, Any]] = []
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """현재 Step의 특성 벡터 반환"""
        obs = self.df.loc[self.current_step, self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        에이전트 행동 수행 및 보상(Reward) 계산
        """
        current_idx = self.current_step
        
        # 매매 수수료 및 슬리피지(어림잡아 거래당 0.05% 감산)
        trading_cost = 0.0005 

        step_reward = 0.0
        profit_loss = 0.0
        actual_return = 0.0
        
        # 행동 평가
        if action > 0 and action <= 6:  # 매수 및 특정 시점 청산
            # Action 1 -> ret_0905, Action 6 -> ret_0930
            target_times = [905, 910, 915, 920, 925, 930]
            action_time = target_times[action - 1]
            target_col = f"ret_{action_time}"
            
            # 타겟 수익률: T일 Feature를 바탕으로 결정한 T+1일 청산 결과
            actual_return = self.df.loc[current_idx, target_col]
            
            # NaN 방지
            if pd.isna(actual_return):
                actual_return = -0.01 # 패널티 줌 (거래 실패 처리)
                
            net_return = actual_return - trading_cost
            step_reward = net_return
            # 자본 복리 계산 연동
            profit_loss = self.balance * net_return
            self.balance = max(0.0, self.balance + profit_loss)
        else:            # 관망(0)
            # 기회비용 패널티 등 복잡한 설정 가능; 현재는 0
            step_reward = 0.0
            profit_loss = 0.0

        # 기록 저장
        self.history.append({
            "step": current_idx,
            "date": self.df.loc[current_idx, "date"] if "date" in self.df.columns else current_idx,
            "action": action,
            "return": actual_return,
            "reward": step_reward,
            "balance": self.balance
        })

        # 다음 스텝 이동
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 파산(Balance <= 0) 시 조기 종료
        if self.balance <= 0:
            terminated = True
            step_reward -= 1.0 # 강한 패널티

        next_obs = self._get_observation() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        
        info = {
            "balance": self.balance,
            "action": action,
            "net_return": step_reward
        }
        
        # Gymnasium v26+: (obs, reward, terminated, truncated, info)
        return next_obs, step_reward, terminated, truncated, info

    def render(self, mode='human'):
        """학습 진행 상황 출력 (디버깅용)"""
        if len(self.history) > 0:
            last = self.history[-1]
            if last["action"] == 0:
                act_str = "SKIP"
            else:
                target_times = [905, 910, 915, 920, 925, 930]
                act_str = f"BUY&SELL_{target_times[int(last['action'])-1]}"
                
            logger.info(f"Step: {last['step']} | Date: {last['date']} | Action: {act_str} | Reward: {last['reward']:.4f} | Balance: {last['balance']:.0f}")


class RLBaselineModel:
    """Stable Baselines3 PPO 알고리즘 기반 단기 매매 모델 래퍼"""
    
    def __init__(self, env: gym.Env):
        # 복수 환경 등 벡터화 대비 DummyVecEnv 통과
        self.env = DummyVecEnv([lambda: env])
        # PPO 모델 초기화 (MlpPolicy: 다층 퍼셉트론)
        self.model = PPO("MlpPolicy", self.env, verbose=0, 
                         learning_rate=0.0003, 
                         n_steps=2048, 
                         batch_size=64, 
                         ent_coef=0.01) # 탐험(Entropy) 계수 부여

    def train(self, total_timesteps: int = 10000):
        """모델 학습 진행"""
        logger.info(f"Starting PPO Agent Training (timesteps: {total_timesteps})...")
        self.model.learn(total_timesteps=total_timesteps)
        logger.info("Training Complete.")

    def evaluate(self) -> pd.DataFrame:
        """현재 학습된 모델을 raw 환경에 직접 적용하여 백테스팅 로그 반환"""
        # DummyVecEnv의 auto-reset이 history를 초기화하므로 raw 환경 직접 사용
        raw_env = self.env.envs[0]
        obs, _ = raw_env.reset()
        done = False
        
        while not done:
            # obs를 batch 차원 추가하여 모델에 전달
            obs_batch = np.expand_dims(obs, axis=0)
            action, _states = self.model.predict(obs_batch, deterministic=True)
            obs, reward, terminated, truncated, info = raw_env.step(int(action[0]))
            done = terminated or truncated
            
        return pd.DataFrame(raw_env.history)
        
    def save(self, model_path: str):
        self.model.save(model_path)
        logger.info(f"RL Baseline Model saved at: {model_path}")
        
    def load(self, model_path: str):
        self.model = PPO.load(model_path, env=self.env)
        logger.info(f"RL Baseline Model loaded from: {model_path}")

if __name__ == "__main__":
    # 간단한 작동 테스트
    logging.basicConfig(level=logging.INFO)
    dummy_data = {
        "date": [20230101, 20230102, 20230103, 20230104, 20230105]*10,
        "feature_1": np.random.randn(50),
        "feature_2": np.random.randn(50),
        "gap_pct": np.random.uniform(-0.05, 0.05, 50),
        "ret_905": np.random.uniform(-0.01, 0.01, 50),
        "ret_910": np.random.uniform(-0.01, 0.02, 50),
        "ret_915": np.random.uniform(-0.02, 0.03, 50),
        "ret_920": np.random.uniform(-0.02, 0.03, 50),
        "ret_925": np.random.uniform(-0.02, 0.03, 50),
        "ret_930": np.random.uniform(-0.03, 0.04, 50)
    }
    df = pd.DataFrame(dummy_data)
    
    env = IntradayTradingEnv(df)
    rl_model = RLBaselineModel(env)
    rl_model.train(total_timesteps=1000)
    
    res = rl_model.evaluate()
    print("Final Balance:", res["balance"].iloc[-1])
