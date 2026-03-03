"""
AutoML_Quant_Trade - 머신러닝 예측 기반 동적 퀀트 전략 모듈
"""
import pandas as pd
from typing import Optional, Dict
from datetime import datetime
from backend.engine.events import MarketEvent, SignalEvent
from backend.strategies.base_strategy import BaseStrategy
from backend.models.predictors import BasePredictor
from backend.models.labeling import extract_triple_barrier_labels
from backend.models.feature_engineer import FeatureEngineer

class MLPredictiveStrategy(BaseStrategy):
    """
    미리 주입된 Predictor(예: XGBoostPredictor)를 통해 종가의 미래 방향성을 추론하고,
    부분 켈리 비율(Quarter-Kelly)로 투자 비중을 결정하는 완전 동적 알고리즘 전략입니다.
    """
    
    def __init__(
        self, 
        predictor: BasePredictor,
        history_manager,
        train_window_days: int = 252,    # 약 1년치 데이터 모인 뒤 초기 학습 시작
        rebalance_freq_days: int = 21,   # 21영업일(약 1달) 주기로 모델 점진적 업데이트
        kelly_fraction: float = 0.25,    # Quarter Kelly 사용 (안전 마진 확보)
        target_prob_threshold: float = 0.55, # 매매 진입 확률 임계치 지정
        avg_win_loss_ratio: float = 1.5, # 켈리 공식에 사용할 b=1.5 가정
        timeframe: str = "daily"
    ):
        super().__init__()
        self.predictor = predictor
        self.history = history_manager # BaseStrategy 밖에서 전체 시계열 접근용 매니저 등
        self.train_window = train_window_days
        self.retrain_freq = rebalance_freq_days
        self.kelly_fraction = kelly_fraction
        self.threshold = target_prob_threshold
        self.b = avg_win_loss_ratio
        
        self.timeframe = timeframe
        self.last_train_idx = 0
        self.event_count = 0
        
        # 특성 추출 캐싱
        self.fe = FeatureEngineer()

    def get_timeframe(self) -> str:
        return self.timeframe

    def _trigger_training(self, ticker: str):
        """과거 전체 데이터를 긁어와서 피처 추출 및 레이블링 후 Predictor에 전달"""
        recent_df = self.history.get_dataframe(ticker, records=self.train_window + self.last_train_idx)
        if len(recent_df) < self.train_window:
            return # 데이터 부족
            
        # 1. 레이블링 (Target 추출) - y
        labeled_df = extract_triple_barrier_labels(
            recent_df, vol_window=20, horizon=5, tp_mult=2.0, sl_mult=1.0
        )
        # FeatureEngineer 형식에 맞춰 소문자 컬럼 및 date 존재 보장
        if 'Close' in recent_df.columns:
            recent_df = recent_df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
        if 'date' not in recent_df.columns:
           recent_df['date'] = recent_df.index
           recent_df.index.name = None

        # 2. 피처 엔지니어링 수행 - X
        feature_df = self.fe.extract(recent_df)
        
        # 3. y와 X 동기화 (결측치 제외)
        # feature_df의 인덱스는 0부터 시작하도록 초기화될 수 있음(extract 내부 reset_index)
        # 따라서 날짜를 기준으로 병합
        labeled_df = labeled_df.reset_index(names=['date']) if 'date' not in labeled_df.columns else labeled_df
        combined = pd.merge(feature_df, labeled_df[['date', 'target_label']], on='date', how='inner').dropna()
        
        if combined.empty:
            return
            
        X = combined.drop(columns=['target_label', 'date'])
        y = combined['target_label']
        
        if not self.predictor.is_fitted():
            # 초기 학습
            self.predictor.fit(X, y)
            print(f"[{datetime.now()}] MLPredictiveStrategy: Initial warm-up complete for {ticker}")
        else:
            # 워크 포워드: 점진적 재학습 (신규 데이터만 넣는게 이상적이지만 예제상 전체 다시 업데이트 처리)
            self.predictor.incremental_fit(X, y)
            print(f"[{datetime.now()}] MLPredictiveStrategy: Continual retrain complete for {ticker}")

    def on_market_data(self, event: MarketEvent) -> Optional[SignalEvent]:
        # 1. 자체 데이터 큐에 적재
        super().on_market_data(event)
        self.event_count += 1
        
        ticker = event.ticker
        
        # 2. Warm-up 체크 및 WFO 주기적 재학습 판별
        if not self.predictor.is_fitted() and self.event_count >= self.train_window:
            self._trigger_training(ticker)
            self.last_train_idx = self.event_count
        elif self.predictor.is_fitted() and (self.event_count - self.last_train_idx) >= self.retrain_freq:
            self._trigger_training(ticker)
            self.last_train_idx = self.event_count
            
        # 모델이 아직 학습 안되었으면 시그널 미발생
        if not self.predictor.is_fitted():
            return None
            
        # 3. 실시간 추론(Inference) 로직
        # 현재 시점 기준 features 산출 (마지막 최신 row 1개)
        # ※ 실제 상용에서는 get_data 한줄만 캐싱하여 1 row 처리하는 등 병목 개선 필요
        recent_df = self.history.get_dataframe(ticker, records=200) # Feature 추출을 위해 충분한 윈도우 필요
        if len(recent_df) < 50:
            return None
            
        if 'Close' in recent_df.columns:
            recent_df = recent_df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
        if 'date' not in recent_df.columns:
           recent_df['date'] = recent_df.index
           recent_df.index.name = None
            
        feat_df = self.fe.extract(recent_df)
        if feat_df.empty:
            return None
            
        # date 컬럼 제외하고 예측에 사용
        latest_X = feat_df.drop(columns=['date']).iloc[[-1]] # 1 row DataFrame

        
        # 예측결과 np.ndarray [[P(-1), P(0), P(1)]]
        probs = self.predictor.predict_proba(latest_X)[0] 
        p_down, p_hold, p_up = probs[0], probs[1], probs[2]
        
        # 4. 방향성 판단
        direction = None
        target_p = 0.0
        if p_up > self.threshold and p_up > p_down:
            direction = "BUY"
            target_p = p_up
        elif p_down > self.threshold and p_down > p_up:
            direction = "SELL"
            target_p = p_down
            
        if not direction:
            # 엣지 우위가 부족하여 베팅 철수/유보(현금화) Signal 발생 (선택적 구현)
            return SignalEvent(
                timestamp=event.timestamp,
                ticker=ticker,
                direction="TARGET",  # "포지션 청산/유보" 의미의 스페셜 타겟
                strength=0.0,      
                strategy_name="MLPredictive"
            )
            
        # 5. Position Sizer (Fractional Kelly) 연산
        # 켈리 공식: f* = p - (q / b) = (bp - q) / b
        # (q = 1-p, b = win_loss_ratio)
        q = 1.0 - target_p
        kelly_f = ((self.b * target_p) - q) / self.b
        
        if kelly_f <= 0:
            return None # 켈리 계산시 오히려 예상 손실이 커 베팅 X
            
        # Quarter-Kelly 적용 (투자 비중 억제)
        final_strength = kelly_f * self.kelly_fraction
        
        # 안전 장치: 단일 베팅이 아무리 높아도 자산의 특정 비중(예: 30%) 초과 금지 
        final_strength = min(final_strength, 0.3)
        
        return SignalEvent(
            timestamp=event.timestamp,
            ticker=ticker,
            direction="TARGET", # 켈리 비중(fraction)은 총 에퀴티의 Target 비중에 가깝기 때문에 TARGET 지시어 사용
            strength=final_strength * (1 if direction == "BUY" else -1), 
            strategy_name="MLPredictive",
            metadata={"p_up": p_up, "p_down": p_down, "kelly_f": kelly_f}
        )
