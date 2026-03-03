"""
AutoML_Quant_Trade - PCA + K-Means 시장 국면 분류기

RobustScaler 정규화 → PCA 차원 축소 → K-Means 군집화
→ 시장 국면(Bull/Bear/Crash) 분류.

Silhouette Score 기반 최적 K 자동 탐색.
"""
import logging
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class RegimeKMeans:
    """PCA + K-Means 시장 국면 분류기"""

    def __init__(self, variance_target: float = 0.95,
                 min_k: int = 2, max_k: int = 5,
                 random_state: int = 42):
        """
        Parameters:
            variance_target: PCA 누적 분산 비율 목표 (0.95 = 95%)
            min_k: 최소 군집 수
            max_k: 최대 군집 수
            random_state: 재현성을 위한 시드
        """
        self.variance_target = variance_target
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.pca = PCA(n_components=variance_target)
        self.kmeans: Optional[KMeans] = None

        self._feature_columns: list = []
        self._optimal_k: int = 0
        self._silhouette_scores: Dict[int, float] = {}
        self._is_fitted: bool = False

    def fit(self, features: pd.DataFrame) -> int:
        """
        특성 데이터로 PCA + K-Means 파이프라인 학습.

        Parameters:
            features: FeatureEngineer.extract()의 결과 (date 컬럼 포함 가능)
        Returns:
            최적 K (군집 수)
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

        # 3. 최적 K 탐색 (Silhouette Score)
        best_k = self.min_k
        best_score = -1

        for k in range(self.min_k, self.max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            self._silhouette_scores[k] = score
            logger.info(f"  K={k}: silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        # 4. 최적 K로 최종 학습
        self._optimal_k = best_k
        self.kmeans = KMeans(
            n_clusters=best_k, random_state=self.random_state, n_init=10
        )
        self.kmeans.fit(X_pca)

        self._is_fitted = True
        logger.info(f"KMeans fitted: optimal K={best_k}, silhouette={best_score:.4f}")

        return best_k

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        특성 데이터의 국면 ID를 예측.

        Parameters:
            features: 새로운 특성 데이터
        Returns:
            국면 ID 배열 (0, 1, ..., K-1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return self.kmeans.predict(X_pca)

    def predict_single(self, features: pd.DataFrame) -> int:
        """단일 행의 국면 ID 예측."""
        return int(self.predict(features)[-1])

    def get_regime_stats(self) -> Dict[int, Dict]:
        """
        각 군집(국면)의 중심점 통계 반환.

        Returns:
            {0: {"center": [...], "count": N}, 1: {...}, ...}
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        stats = {}
        labels = self.kmeans.labels_
        for k in range(self._optimal_k):
            mask = labels == k
            stats[k] = {
                "center": self.kmeans.cluster_centers_[k].tolist(),
                "count": int(mask.sum()),
                "ratio": float(mask.sum() / len(labels)),
            }
        return stats

    @property
    def optimal_k(self) -> int:
        return self._optimal_k

    @property
    def silhouette_scores(self) -> Dict[int, float]:
        return self._silhouette_scores

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
        path = path or os.path.join(Settings.MODEL_DIR, "regime_kmeans.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "kmeans": self.kmeans,
                "feature_columns": self._feature_columns,
                "optimal_k": self._optimal_k,
                "silhouette_scores": self._silhouette_scores,
            }, f)
        logger.info(f"KMeans model saved to {path}")

    def load(self, path: str = None):
        """저장된 모델 로드."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "regime_kmeans.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.kmeans = data["kmeans"]
        self._feature_columns = data["feature_columns"]
        self._optimal_k = data["optimal_k"]
        self._silhouette_scores = data["silhouette_scores"]
        self._is_fitted = True
        logger.info(f"KMeans model loaded from {path}")
