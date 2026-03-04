"""
AutoML_Quant_Trade - Gaussian HMM 시장 국면 감지기

Hidden Markov Model을 사용하여 시장 상태(Bull/Bear/Crash)의
시계열 전이를 모델링.

■ 미래참조 편향 방지:
  walk_forward_predict()를 통해 t 시점 예측에
  반드시 [t-train_window, t-1] 구간만 사용.
"""
import logging
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

from backend.config.settings import Settings
from backend.models.pytorch_hmm import TorchGaussianHMM

logger = logging.getLogger(__name__)


class RegimeHMM:
    """Gaussian HMM 시계열 국면 감지기"""

    def __init__(self, n_regimes: int = None,
                 pca_variance: float = 0.90,
                 random_state: int = 42):
        """
        Parameters:
            n_regimes: HMM 상태(국면) 수 (기본: Settings.REGIME_COUNT)
            pca_variance: PCA 누적 분산 목표
            random_state: 재현성 시드
        """
        self.n_regimes = n_regimes or Settings.REGIME_COUNT
        self.pca_variance = pca_variance
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.pca = PCA(n_components=pca_variance)
        self.model: Optional[GaussianHMM] = None

        self._feature_columns: list = []
        self._is_fitted: bool = False

    def fit(self, features: pd.DataFrame, n_iter: int = 100):
        """
        특성 데이터로 Gaussian HMM 학습.

        Parameters:
            features: FeatureEngineer.extract() 결과
            n_iter: EM 알고리즘 최대 반복 횟수
        """
        X = self._prepare_features(features)

        # 정규화 + PCA
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)

        n_components_pca = self.pca.n_components_
        logger.info(
            f"HMM PCA: {X.shape[1]} → {n_components_pca} components "
            f"(variance: {sum(self.pca.explained_variance_ratio_):.3f})"
        )

        use_pytorch = getattr(Settings, 'USE_PYTORCH_HMM', True)
        if use_pytorch:
            self.model = TorchGaussianHMM(
                n_components=self.n_regimes,
                n_features=n_components_pca,
                n_iter=n_iter,
                random_state=self.random_state,
            )
        else:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                random_state=self.random_state,
            )
            
        self.model.fit(X_pca)

        self._is_fitted = True

        # 전이 확률 행렬 로깅
        logger.info(f"HMM fitted with {self.n_regimes} regimes")
        logger.info(f"Transition matrix:\n{np.round(self.model.transmat_, 3)}")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        각 시점의 국면 확률 벡터 γ(t) 반환.

        Returns:
            shape=(n_samples, n_regimes) 확률 행렬
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return self.model.predict_proba(X_pca)

    def decode(self, features: pd.DataFrame) -> np.ndarray:
        """
        Viterbi 알고리즘으로 최적 국면 시퀀스 추정.

        Returns:
            국면 ID 배열 (0, 1, ..., n_regimes-1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        _, states = self.model.decode(X_pca)
        return states

    def walk_forward_predict(self, features: pd.DataFrame,
                              train_window: int = 252,
                              retrain_freq: int = 21) -> pd.DataFrame:
        """
        Walk-Forward 국면 예측 — 미래참조 편향 완전 방지.

        t 시점의 국면 예측에는 반드시 [t-train_window, t-1] 구간의 데이터만 사용.
        retrain_freq일마다 모델을 재학습하여 최신 시장 상태 반영.

        Parameters:
            features: 전체 특성 데이터 (date 컬럼 포함)
            train_window: 학습 윈도우 크기 (일 수)
            retrain_freq: 재학습 주기 (일 수)

        Returns:
            DataFrame: [date, regime, prob_0, prob_1, prob_2]

        ■ 미래참조 편향 방지:
          - t 시점 모델은 [t-train_window, t-1] 데이터로만 학습
          - t 시점 예측은 학습된 모델에 t 시점 특성만 입력
          - t+1 이후 데이터는 절대 접근 불가
        """
        dates = features["date"].values if "date" in features.columns else np.arange(len(features))
        X_full = self._prepare_features(features)

        results = []

        for t in range(train_window, len(X_full)):
            # 재학습 시점 체크
            if (t - train_window) % retrain_freq == 0:
                # [t-train_window, t-1] 구간으로만 학습
                train_data = X_full[t - train_window: t]
                self._fit_internal(train_data)

            # t 시점 단일 예측
            t_data = X_full[t: t + 1]
            t_scaled = self.scaler.transform(t_data)
            t_pca = self.pca.transform(t_scaled)

            proba = self.model.predict_proba(t_pca)[0]
            regime = int(np.argmax(proba))

            row = {"date": dates[t], "regime": regime}
            for r in range(self.n_regimes):
                row[f"prob_{r}"] = float(proba[r])
            results.append(row)

        result_df = pd.DataFrame(results)
        logger.info(
            f"Walk-forward prediction: {len(result_df)} points "
            f"(train_window={train_window}, retrain_freq={retrain_freq})"
        )

        return result_df

    def _fit_internal(self, X: np.ndarray):
        """내부용 학습 (Walk-Forward에서 호출)."""
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)

        use_pytorch = getattr(Settings, 'USE_PYTORCH_HMM', True)
        if use_pytorch:
            self.model = TorchGaussianHMM(
                n_components=self.n_regimes,
                n_features=X_pca.shape[1],
                n_iter=50,
                random_state=self.random_state,
            )
        else:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=50,  # Walk-Forward에서는 빠른 수렴 우선
                random_state=self.random_state,
            )
        self.model.fit(X_pca)
        self._is_fitted = True

    def get_transition_matrix(self) -> np.ndarray:
        """전이 확률 행렬 반환."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.transmat_

    def get_stationary_distribution(self) -> np.ndarray:
        """정상 분포(stationary distribution) 반환."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        # 고유벡터로 정상 분포 계산
        eigenvalues, eigenvectors = np.linalg.eig(self.model.transmat_.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return stationary

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """date 컬럼 제거 후 수치 배열 변환."""
        df = features.copy()
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        if not self._feature_columns:
            self._feature_columns = df.columns.tolist()

        return df.values

    def save(self, path: str = None):
        """모델 저장."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "regime_hmm.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "model": self.model,
                "feature_columns": self._feature_columns,
                "n_regimes": self.n_regimes,
            }, f)
        logger.info(f"HMM model saved to {path}")

    def load(self, path: str = None):
        """모델 로드."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "regime_hmm.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.model = data["model"]
        self._feature_columns = data["feature_columns"]
        self.n_regimes = data["n_regimes"]
        self._is_fitted = True
        logger.info(f"HMM model loaded from {path}")
