"""
AutoML_Quant_Trade - Gaussian Mixture Model 시장 국면 분류기

GMM을 사용하여 소프트 클러스터링(확률적 소속) 기반 국면 탐지.
BIC/AIC 기반 최적 컴포넌트 수 자동 탐색.

■ RegimeKMeans와 동일 인터페이스:
  fit(), predict(), predict_proba(), save(), load()
"""
import logging
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class RegimeGMM:
    """Gaussian Mixture Model 기반 시장 국면 분류기"""

    def __init__(self, min_k: int = 2, max_k: int = 5,
                 covariance_type: str = "full",
                 variance_target: float = 0.95,
                 random_state: int = 42):
        """
        Parameters:
            min_k: 최소 컴포넌트 수
            max_k: 최대 컴포넌트 수
            covariance_type: 공분산 유형 ('full', 'tied', 'diag', 'spherical')
            variance_target: PCA 누적 분산 비율 목표
            random_state: 재현성 시드
        """
        self.min_k = min_k
        self.max_k = max_k
        self.covariance_type = covariance_type
        self.variance_target = variance_target
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.pca = PCA(n_components=variance_target)
        self.gmm: Optional[GaussianMixture] = None

        self._feature_columns: list = []
        self._optimal_k: int = 0
        self._bic_scores: Dict[int, float] = {}
        self._aic_scores: Dict[int, float] = {}
        self._is_fitted: bool = False

    def fit(self, features: pd.DataFrame) -> int:
        """
        특성 데이터로 PCA + GMM 파이프라인 학습.

        Parameters:
            features: FeatureEngineer.extract()의 결과 (date 컬럼 포함 가능)
        Returns:
            최적 K (컴포넌트 수)
        """
        X = self._prepare_features(features)

        # 1. 정규화
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA 차원 축소
        X_pca = self.pca.fit_transform(X_scaled)
        n_components = self.pca.n_components_
        explained_var = sum(self.pca.explained_variance_ratio_)
        logger.info(
            f"PCA: {X.shape[1]} features → {n_components} components "
            f"(explained variance: {explained_var:.3f})"
        )

        # 3. 최적 K 탐색 (BIC 기준, AIC도 기록)
        best_k = self.min_k
        best_bic = np.inf

        for k in range(self.min_k, self.max_k + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=3,
                max_iter=200,
            )
            gm.fit(X_pca)
            bic = gm.bic(X_pca)
            aic = gm.aic(X_pca)
            self._bic_scores[k] = bic
            self._aic_scores[k] = aic
            logger.info(f"  K={k}: BIC={bic:.1f}, AIC={aic:.1f}")

            if bic < best_bic:
                best_bic = bic
                best_k = k

        # 4. 최적 K로 최종 학습
        self._optimal_k = best_k
        self.gmm = GaussianMixture(
            n_components=best_k,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
            max_iter=300,
        )
        self.gmm.fit(X_pca)

        self._is_fitted = True
        logger.info(f"GMM fitted: optimal K={best_k}, BIC={best_bic:.1f}")

        return best_k

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        특성 데이터의 국면 ID를 예측.

        Returns:
            국면 ID 배열 (0, 1, ..., K-1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return self.gmm.predict(X_pca)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        각 시점의 국면 확률 벡터 반환 (소프트 클러스터링).

        Returns:
            shape=(n_samples, n_components) 확률 행렬
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return self.gmm.predict_proba(X_pca)

    def predict_single(self, features: pd.DataFrame) -> int:
        """단일 행의 국면 ID 예측."""
        return int(self.predict(features)[-1])

    def get_regime_stats(self) -> Dict[int, Dict]:
        """
        각 컴포넌트(국면)의 평균, 공분산 통계 반환.

        Returns:
            {0: {"mean": [...], "weight": float}, 1: {...}, ...}
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        stats = {}
        for k in range(self._optimal_k):
            stats[k] = {
                "mean": self.gmm.means_[k].tolist(),
                "weight": float(self.gmm.weights_[k]),
            }
        return stats

    @property
    def optimal_k(self) -> int:
        return self._optimal_k

    @property
    def bic_scores(self) -> Dict[int, float]:
        return self._bic_scores

    @property
    def aic_scores(self) -> Dict[int, float]:
        return self._aic_scores

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """date 컬럼 제거 후 수치 배열 변환."""
        df = features.copy()
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        if not self._feature_columns:
            self._feature_columns = df.columns.tolist()

        return df.values

    def save(self, path: str = None):
        """학습된 모델을 파일로 저장."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "regime_gmm.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "gmm": self.gmm,
                "feature_columns": self._feature_columns,
                "optimal_k": self._optimal_k,
                "bic_scores": self._bic_scores,
                "aic_scores": self._aic_scores,
                "covariance_type": self.covariance_type,
            }, f)
        logger.info(f"GMM model saved to {path}")

    def load(self, path: str = None):
        """저장된 모델 로드."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "regime_gmm.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.gmm = data["gmm"]
        self._feature_columns = data["feature_columns"]
        self._optimal_k = data["optimal_k"]
        self._bic_scores = data["bic_scores"]
        self._aic_scores = data["aic_scores"]
        self.covariance_type = data.get("covariance_type", "full")
        self._is_fitted = True
        logger.info(f"GMM model loaded from {path}")
