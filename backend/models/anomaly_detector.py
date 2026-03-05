"""
AutoML_Quant_Trade - 비지도 이상 탐지 모듈

IsolationForest + PyTorch Autoencoder 앙상블로
시장 급변(이상치) 구간을 탐지.

■ IsolationForest: 빠른 이상 스코어, 트리 기반
■ Autoencoder: 재구성 오류(reconstruction error) 기반 탐지
■ score(): 두 모델의 앙상블 이상 스코어 [0, 1] 반환
"""
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from backend.config.settings import Settings

logger = logging.getLogger(__name__)

# ── PyTorch Autoencoder (선택적 GPU 지원) ──
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Autoencoder will be disabled.")


class _Autoencoder(nn.Module):
    """대칭 구조의 Denoising Autoencoder."""

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (대칭)
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AnomalyDetector:
    """비지도 이상 탐지 앙상블 (IsolationForest + Autoencoder)"""

    def __init__(self, contamination: float = 0.05,
                 ae_hidden_dims: list = None,
                 ae_epochs: int = 50,
                 ae_lr: float = 1e-3,
                 ae_weight: float = 0.5,
                 random_state: int = 42):
        """
        Parameters:
            contamination: IsolationForest 이상치 비율 추정
            ae_hidden_dims: Autoencoder 히든 레이어 차원 리스트
            ae_epochs: Autoencoder 학습 에폭
            ae_lr: Autoencoder 학습률
            ae_weight: Autoencoder 스코어 가중치 (0~1, 나머지는 IF)
            random_state: 재현성 시드
        """
        self.contamination = contamination
        self.ae_hidden_dims = ae_hidden_dims or [32, 16]
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_weight = ae_weight
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.iforest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1,
        )

        self._autoencoder: Optional[_Autoencoder] = None
        self._ae_threshold: float = 0.0  # 재구성 오류 임계치
        self._is_fitted: bool = False
        self._feature_columns: list = []
        self._device = "cpu"

    def fit(self, features: pd.DataFrame):
        """
        IsolationForest + Autoencoder 학습.

        Parameters:
            features: FeatureEngineer.extract() 결과
        """
        X = self._prepare_features(features)
        X_scaled = self.scaler.fit_transform(X)

        # 1. IsolationForest 학습
        self.iforest.fit(X_scaled)
        logger.info(f"IsolationForest fitted: {X_scaled.shape[0]} samples, "
                     f"contamination={self.contamination}")

        # 2. Autoencoder 학습 (PyTorch 사용 가능 시)
        if TORCH_AVAILABLE:
            self._fit_autoencoder(X_scaled)
        else:
            logger.info("Autoencoder skipped (PyTorch unavailable)")

        self._is_fitted = True

    def _fit_autoencoder(self, X_scaled: np.ndarray):
        """PyTorch Autoencoder 학습."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = str(device)

        input_dim = X_scaled.shape[1]
        self._autoencoder = _Autoencoder(input_dim, self.ae_hidden_dims).to(device)
        optimizer = torch.optim.Adam(self._autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        
        # BatchNorm1d requires at least 2 samples per batch. 
        # If total samples < 2, we can't train with BN.
        if len(X_tensor) < 2:
            logger.warning("Not enough samples for Autoencoder training (min 2 required for BatchNorm)")
            return

        batch_size = min(256, len(X_tensor))
        # drop_last=True ensures we don't get a final batch of size 1
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=(len(X_tensor) > batch_size)
        )

        self._autoencoder.train()
        for epoch in range(self.ae_epochs):
            total_loss = 0.0
            for (batch,) in loader:
                # Denoising: 입력에 작은 노이즈 추가
                noisy = batch + 0.05 * torch.randn_like(batch)
                output = self._autoencoder(noisy)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(X_tensor)
                logger.info(f"  AE epoch {epoch+1}/{self.ae_epochs}: loss={avg_loss:.6f}")

        # 임계치 설정: 학습 데이터의 재구성 오류 분포에서 상위 contamination%
        self._autoencoder.eval()
        with torch.no_grad():
            recon = self._autoencoder(X_tensor)
            errors = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        self._ae_threshold = float(np.percentile(errors, (1 - self.contamination) * 100))
        logger.info(f"Autoencoder fitted: threshold={self._ae_threshold:.6f}, device={device}")

    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        이상치 스코어 반환 [0, 1]. 1에 가까울수록 이상.

        Parameters:
            features: 피처 DataFrame
        Returns:
            이상치 스코어 Series (index = features.index)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X)

        # IsolationForest 스코어: decision_function → 낮을수록 이상
        if_raw = self.iforest.decision_function(X_scaled)
        # 정규화: [-1, 1] 범위를 [0, 1]로 (낮은 값 → 높은 이상 스코어)
        if_score = 1.0 - ((if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-10))

        # Autoencoder 스코어
        if self._autoencoder is not None and TORCH_AVAILABLE:
            device = torch.device(self._device)
            self._autoencoder.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                recon = self._autoencoder(X_tensor)
                ae_errors = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
            # 정규화: 임계치 기준으로 [0, 1] 스케일
            ae_score = np.clip(ae_errors / (self._ae_threshold + 1e-10), 0.0, 2.0) / 2.0

            # 앙상블
            final_score = (1 - self.ae_weight) * if_score + self.ae_weight * ae_score
        else:
            final_score = if_score

        return pd.Series(final_score, index=features.index, name="anomaly_score")

    def is_anomaly(self, features: pd.DataFrame, threshold: float = 0.7) -> pd.Series:
        """
        이상치 여부 부울 Series.

        Parameters:
            features: 피처 DataFrame
            threshold: 이상치 판정 임계치 (기본 0.7)
        """
        return self.score(features) > threshold

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
        path = path or os.path.join(Settings.MODEL_DIR, "anomaly_detector.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Autoencoder state_dict 분리 저장
        ae_state = None
        if self._autoencoder is not None:
            ae_state = self._autoencoder.state_dict()

        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "iforest": self.iforest,
                "ae_state": ae_state,
                "ae_hidden_dims": self.ae_hidden_dims,
                "ae_threshold": self._ae_threshold,
                "feature_columns": self._feature_columns,
                "contamination": self.contamination,
                "device": self._device,
            }, f)
        logger.info(f"AnomalyDetector saved to {path}")

    def load(self, path: str = None):
        """모델 로드."""
        import os
        path = path or os.path.join(Settings.MODEL_DIR, "anomaly_detector.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.scaler = data["scaler"]
        self.iforest = data["iforest"]
        self._feature_columns = data["feature_columns"]
        self._ae_threshold = data["ae_threshold"]
        self.ae_hidden_dims = data["ae_hidden_dims"]
        self.contamination = data["contamination"]
        self._device = data.get("device", "cpu")

        # Autoencoder 복원
        if data["ae_state"] is not None and TORCH_AVAILABLE:
            input_dim = len(self._feature_columns)
            self._autoencoder = _Autoencoder(input_dim, self.ae_hidden_dims)
            self._autoencoder.load_state_dict(data["ae_state"])
            self._autoencoder.eval()
            logger.info("Autoencoder state restored.")

        self._is_fitted = True
        logger.info(f"AnomalyDetector loaded from {path}")
