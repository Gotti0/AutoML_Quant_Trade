"""
AutoML_Quant_Trade - 종목 간 비지도학습 클러스터 분석 모듈

여러 종목의 피처 벡터를 PCA → HDBSCAN 파이프라인으로 군집화하고,
군집 내 상대 강도(relative strength) 순위를 계산.

■ 스크리너의 핵심 엔진:
  - 종목 간 유사성 기반 그룹화
  - 군집별 특성 프로파일 자동 라벨링
  - 군집 내 모멘텀 순위 산출
"""
import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from backend.config.settings import Settings

logger = logging.getLogger(__name__)

# HDBSCAN은 선택적 의존성
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("hdbscan not installed. Using sklearn KMeans fallback.")
    from sklearn.cluster import KMeans

# FAISS 가속 (선택적 의존성)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("faiss not installed. FAISS acceleration disabled.")


class CrossAssetClusterAnalyzer:
    """종목 간 비지도학습 클러스터 분석"""

    def __init__(self, n_components_pca: float = 0.95,
                 min_cluster_size: int = 5,
                 min_samples: int = 3,
                 fallback_k: int = 5,
                 use_faiss: bool = True,
                 random_state: int = 42):
        """
        Parameters:
            n_components_pca: PCA 누적 분산 비율 목표
            min_cluster_size: HDBSCAN 최소 군집 크기
            min_samples: HDBSCAN 코어 포인트 최소 이웃 수
            fallback_k: HDBSCAN 미사용 시 KMeans 군집 수
            random_state: 재현성 시드
        """
        self.n_components_pca = n_components_pca
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.fallback_k = fallback_k
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_components_pca)
        self._clusterer = None

        self._ticker_list: List[str] = []
        self._labels: Optional[np.ndarray] = None
        self._feature_matrix: Optional[np.ndarray] = None
        self._ticker_features: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    def fit(self, multi_asset_features: Dict[str, pd.DataFrame]):
        """
        여러 종목의 피처를 종합하여 군집화.

        Parameters:
            multi_asset_features: {ticker: FeatureEngineer.extract() 결과}
                각 DataFrame은 해당 종목의 시계열 피처.
                종목별로 최신 N일 통계 요약 벡터를 산출하여 하나의 행으로 축약.
        """
        # 1. 종목별 요약 피처벡터 생성 (시계열 → 단면)
        summary_rows = []
        tickers = []

        for ticker, feat_df in multi_asset_features.items():
            if feat_df.empty or len(feat_df) < 20:
                continue

            summary = self._summarize_features(feat_df)
            if summary is not None:
                summary_rows.append(summary)
                tickers.append(ticker)

        if len(summary_rows) < 2:
            logger.warning(f"Not enough tickers for clustering: {len(summary_rows)}")
            self._is_fitted = False
            return

        self._ticker_list = tickers
        summary_df = pd.DataFrame(summary_rows, index=tickers)
        # NaN/Inf 처리
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._ticker_features = summary_df

        # 2. 정규화 + PCA
        X_scaled = self.scaler.fit_transform(summary_df.values)
        X_pca = self.pca.fit_transform(X_scaled)
        n_comp = self.pca.n_components_
        logger.info(f"PCA: {summary_df.shape[1]} features → {n_comp} components")

        # 3. 군집화
        if FAISS_AVAILABLE and self.use_faiss:
            self._labels = self._faiss_cluster(X_pca)
            n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
            logger.info(f"FAISS KMeans: {n_clusters} clusters")
        elif HDBSCAN_AVAILABLE:
            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
            )
            self._labels = self._clusterer.fit_predict(X_pca)
            n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
            n_noise = (self._labels == -1).sum()
            logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        else:
            self._clusterer = KMeans(
                n_clusters=self.fallback_k,
                random_state=self.random_state,
                n_init=10,
            )
            self._labels = self._clusterer.fit_predict(X_pca)
            logger.info(f"KMeans fallback: {self.fallback_k} clusters")

        self._feature_matrix = X_pca
        self._is_fitted = True

    def _faiss_cluster(self, X_pca: np.ndarray) -> np.ndarray:
        """
        FAISS KMeans를 사용한 고속 클러스터링.
        
        FAISS의 KMeans는 C++ 백엔드로 거리 행렬 계산을 수행하여
        scikit-learn 대비 대규모 데이터에서 10~50x 빠른 속도.
        """
        n_samples = X_pca.shape[0]
        k = self.fallback_k
        d = X_pca.shape[1]
        
        # float32로 변환 (FAISS 요구사항)
        X_f32 = np.ascontiguousarray(X_pca, dtype=np.float32)
        
        # FAISS KMeans
        kmeans = faiss.Kmeans(
            d, k,
            niter=20,
            verbose=False,
            seed=self.random_state,
        )
        kmeans.train(X_f32)
        
        # 가장 가까운 centroid 할당
        _, labels = kmeans.index.search(X_f32, 1)
        return labels.flatten()

    def _summarize_features(self, feat_df: pd.DataFrame) -> Optional[Dict]:
        """
        시계열 피처 DataFrame을 단일 요약 벡터로 축약.

        마지막 21일 기준 평균/표준편차/최신값을 결합하여
        종목의 현재 상태를 요약.
        """
        df = feat_df.copy()
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        if len(df) < 10:
            return None

        recent = df.tail(21)
        summary = {}

        for col in df.columns:
            summary[f"{col}_mean"] = recent[col].mean()
            summary[f"{col}_std"] = recent[col].std()
            summary[f"{col}_last"] = df[col].iloc[-1]

        return summary

    def get_clusters(self) -> Dict[int, List[str]]:
        """
        군집 할당 결과 반환.

        Returns:
            {cluster_id: [ticker, ...]}
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        clusters: Dict[int, List[str]] = {}
        for ticker, label in zip(self._ticker_list, self._labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(ticker)

        return clusters

    def get_ticker_cluster(self, ticker: str) -> int:
        """특정 종목의 군집 ID 반환."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        if ticker not in self._ticker_list:
            return -1
        idx = self._ticker_list.index(ticker)
        return int(self._labels[idx])

    def rank_within_cluster(self, cluster_id: int,
                             rank_feature: str = "return_21d_last") -> pd.DataFrame:
        """
        군집 내 종목들의 상대 강도 순위 반환.

        Parameters:
            cluster_id: 순위를 매길 군집 ID
            rank_feature: 순위 기준 피처 (기본: 21일 수익률)
        Returns:
            DataFrame[ticker, score, rank] — score 내림차순
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")

        clusters = self.get_clusters()
        tickers_in_cluster = clusters.get(cluster_id, [])

        if not tickers_in_cluster or self._ticker_features is None:
            return pd.DataFrame(columns=["ticker", "score", "rank"])

        # 해당 피처가 없으면 첫 번째 가용 피처 사용
        available_cols = self._ticker_features.columns.tolist()
        if rank_feature not in available_cols:
            rank_feature = available_cols[0] if available_cols else None
            if rank_feature is None:
                return pd.DataFrame(columns=["ticker", "score", "rank"])

        scores = []
        for t in tickers_in_cluster:
            if t in self._ticker_features.index:
                scores.append({
                    "ticker": t,
                    "score": float(self._ticker_features.loc[t, rank_feature]),
                })

        df = pd.DataFrame(scores)
        if df.empty:
            return df

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def get_cluster_profile(self, cluster_id: int) -> Dict:
        """
        군집의 특성 프로파일 반환.

        Returns:
            {"count": N, "mean_features": {...}, "label_hint": str}
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")

        clusters = self.get_clusters()
        tickers_in_cluster = clusters.get(cluster_id, [])

        if not tickers_in_cluster or self._ticker_features is None:
            return {"count": 0, "mean_features": {}, "label_hint": "unknown"}

        cluster_df = self._ticker_features.loc[
            self._ticker_features.index.isin(tickers_in_cluster)
        ]

        mean_features = cluster_df.mean().to_dict()

        # 자동 라벨링 힌트 (모멘텀 vs 방어 vs 고변동성)
        momentum_key = None
        vol_key = None
        for col in mean_features:
            if "return_21d_mean" in col:
                momentum_key = col
            if "vol_21d_mean" in col:
                vol_key = col

        label = "neutral"
        if momentum_key and vol_key:
            mom_val = mean_features.get(momentum_key, 0)
            vol_val = mean_features.get(vol_key, 0)
            if mom_val > 0.02 and vol_val < 0.3:
                label = "momentum_low_vol"
            elif mom_val > 0.02:
                label = "momentum_high_vol"
            elif mom_val < -0.02:
                label = "bearish"
            elif vol_val < 0.15:
                label = "defensive"
            else:
                label = "neutral"

        return {
            "count": len(tickers_in_cluster),
            "mean_features": {k: round(v, 6) for k, v in list(mean_features.items())[:10]},
            "label_hint": label,
        }

    @property
    def n_clusters(self) -> int:
        if self._labels is None:
            return 0
        return len(set(self._labels)) - (1 if -1 in self._labels else 0)

    def save(self, path: str = None):
        """모델 저장."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "cluster_analyzer.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "clusterer": self._clusterer,
                "ticker_list": self._ticker_list,
                "labels": self._labels,
                "ticker_features": self._ticker_features,
                "feature_matrix": self._feature_matrix,
            }, f)
        logger.info(f"ClusterAnalyzer saved to {path}")

    def load(self, path: str = None):
        """모델 로드."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "cluster_analyzer.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self._clusterer = data["clusterer"]
        self._ticker_list = data["ticker_list"]
        self._labels = data["labels"]
        self._ticker_features = data["ticker_features"]
        self._feature_matrix = data["feature_matrix"]
        self._is_fitted = True
        logger.info(f"ClusterAnalyzer loaded from {path}")
