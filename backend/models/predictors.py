"""
AutoML_Quant_Trade - 모델 추상화 레이어

향후 Google AutoML(Vertex AI) 등 Cloud ML로의 확장을 위해, 시그널 추론 파트를
전략 객체로부터 Decoupling(의존성 분리)하는 역할을 담당합니다.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import xgboost as xgb

class BasePredictor(ABC):
    """모든 ML/DL 예측기가 구현해야 할 인터페이스"""
    
    @abstractmethod
    def is_fitted(self) -> bool:
        """모델이 초기 훈련(Warm-up)을 마쳤는지 여부 반환"""
        pass
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """초기 다량의 데이터를 기반으로 모델 훈련을 수행"""
        pass
        
    @abstractmethod
    def incremental_fit(self, X: pd.DataFrame, y: pd.Series):
        """기존 가중치를 보존한 채 신규 데이터를 점진적으로(Rolling) 추가 학습"""
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        특성 벡터 X를 받아 클래스별 확률 배열 반환
        Target 레이블 규칙: -1(하락), 0(만료/유지), 1(상승)
        반환 예시: np.array([[P(-1), P(0), P(1)]])
        """
        pass

class XGBoostPredictor(BasePredictor):
    """
    로컬 백테스팅 및 실거래용 XGBoost 분류기 래퍼.
    점진적 학습(Incremental Learning)인 WFO를 지원.
    """
    
    def __init__(self, **xgb_params):
        # 기본 하이퍼파라미터
        self.params = {
            'objective': 'multi:softprob', # 다중 클래스 확률
            'num_class': 3,                # -1, 0, 1
            'eval_metric': 'mlogloss',
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',         # 속도 최적화
            'random_state': 42
        }
        self.params.update(xgb_params)
        self.booster: xgb.Booster = None
        
    def _map_labels(self, y: pd.Series) -> pd.Series:
        """XGBoost는 클래스 레이블이 0부터 시작해야 함. -1, 0, 1 -> 0, 1, 2 로 변환"""
        mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
        return y.map(mapping)

    def is_fitted(self) -> bool:
        return self.booster is not None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Warming up: 처음 모델 생성"""
        y_mapped = self._map_labels(y)
        # 결측치가 있는 target 행 제거 보장
        mask = y_mapped.notna()
        dtrain = xgb.DMatrix(X[mask], label=y_mapped[mask])
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=100
        )

    def incremental_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Walk-Forward Optimization.
        미리 훈련된 booster를 재활용(xgb_model 인자)하여 새로운 트리를 추가 생성.
        """
        if not self.is_fitted():
            return self.fit(X, y)
            
        y_mapped = self._map_labels(y)
        mask = y_mapped.notna()
        dtrain = xgb.DMatrix(X[mask], label=y_mapped[mask])
        
        # xgb_model 에 이전 self.booster 를 던져주어 이어서 훈련
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=30, # 추가 훈련은 round를 작게 세팅
            xgb_model=self.booster
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted():
            raise ValueError("Model is not fitted yet.")
        dtest = xgb.DMatrix(X)
        return self.booster.predict(dtest)
