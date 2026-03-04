"""
AutoML_Quant_Trade - PyTorch 기반 Gaussian HMM

Gradient Descent 방식으로 NLL(Negative Log-Likelihood)을 최소화하여 
HMM(가우시안 믹스처)의 은닉 국면을 학습하는 커스텀 모듈.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

logger = logging.getLogger(__name__)

class TorchGaussianHMM(nn.Module):
    """
    PyTorch를 활용한 Gaussian Hidden Markov Model 
    GPU(CUDA) 연산을 지원하여 학습 속도를 가속화.
    """
    def __init__(self, n_components=3, n_features=1, n_iter=100, random_state=42, device=None):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_iter = n_iter
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        torch.manual_seed(random_state)
        
        # 전이 확률 (정규화 전 log-probs)
        self.unnormalized_transmat = nn.Parameter(torch.randn(n_components, n_components))
        # 초기 국면 확률 (정규화 전 log-probs)
        self.unnormalized_startprob = nn.Parameter(torch.randn(n_components))
        # 평균 벡터
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        
        # 공분산 행렬 (Cholesky Factor) - 양의 정부호 보장
        covs = torch.stack([torch.eye(n_features) for _ in range(n_components)])
        self.unnormalized_cov_chol = nn.Parameter(torch.linalg.cholesky(covs))
        
        self.transmat_ = None

    @property
    def transmat(self):
        """정규화된 전이 확률 행렬"""
        return F.softmax(self.unnormalized_transmat, dim=1)

    @property
    def startprob(self):
        """정규화된 초기 국면 확률"""
        return F.softmax(self.unnormalized_startprob, dim=0)

    def _get_covs(self):
        """Cholesky 분해로 양의 정부호(Positive Definite) 공분산 행렬 계산"""
        L = torch.tril(self.unnormalized_cov_chol)          # 하삼각 행렬 추출
        covs = torch.matmul(L, L.transpose(-1, -2))         # L * L^T
        jitter = torch.eye(self.n_features, device=self.device) * 1e-4  # 안정성용 jitter
        return covs + jitter

    def _emission_log_prob(self, X):
        """
        다변량 가우시안 발현 로그 확률 계산
        X: (T, F), 반환: (T, N)
        """
        covs = self._get_covs()
        emissions = []
        for i in range(self.n_components):
            dist = torch.distributions.MultivariateNormal(self.means[i], covs[i])
            emissions.append(dist.log_prob(X))
        return torch.stack(emissions, dim=1)

    def forward_algorithm(self, X):
        """Forward(전향) 패스를 통해 각 상태별 alpha(로그 확률) 계산"""
        T = X.shape[0]
        log_e = self._emission_log_prob(X) 
        log_transmat = torch.log(self.transmat + 1e-8)
        
        alpha = torch.zeros(T, self.n_components, device=self.device)
        alpha[0] = torch.log(self.startprob + 1e-8) + log_e[0]
        
        for t in range(1, T):
            step = alpha[t-1].unsqueeze(1) + log_transmat
            alpha[t] = torch.logsumexp(step, dim=0) + log_e[t]
            
        log_prob = torch.logsumexp(alpha[T-1], dim=0)
        return log_prob, alpha

    def fit(self, X_numpy):
        """그래디언트 디센트를 사용한 NLL 최소화 학습"""
        self.to(self.device)
        X = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        
        optimizer = optim.Adam(self.parameters(), lr=0.02)
        
        for idx in range(self.n_iter):
            optimizer.zero_grad()
            log_prob, _ = self.forward_algorithm(X)
            loss = -log_prob / X.shape[0]  # 평균 Negative Log-Likelihood
            
            loss.backward()
            optimizer.step()
            
        self.transmat_ = self.transmat.detach().cpu().numpy()
        logger.debug(f"[TorchGaussianHMM] Fited on device {self.device}. Initial Loss = N/A, Final Loss = {loss.item():.4f}")

    def predict_proba(self, X_numpy):
        """
        Forward-Backward(전향-후향) 알고리즘을 사용한 국면(State) 확률 추론
        반환: (T, N) 
        """
        self.to(self.device)
        X = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_prob, alpha = self.forward_algorithm(X)
            
            T = X.shape[0]
            log_e = self._emission_log_prob(X)
            log_transmat = torch.log(self.transmat + 1e-8)
            
            beta = torch.zeros(T, self.n_components, device=self.device)
            beta[T-1] = 0.0
            
            for t in range(T-2, -1, -1):
                step = log_transmat + log_e[t+1].unsqueeze(0) + beta[t+1].unsqueeze(0)
                beta[t] = torch.logsumexp(step, dim=1)
                
            gamma_num = alpha + beta
            gamma_den = torch.logsumexp(gamma_num, dim=1, keepdim=True)
            gamma = torch.exp(gamma_num - gamma_den)
            
            return gamma.cpu().numpy()

    def decode(self, X_numpy):
        """Viterbi 알고리즘 최상경로(국면) 추정"""
        self.to(self.device)
        X = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            T = X.shape[0]
            log_e = self._emission_log_prob(X)
            log_transmat = torch.log(self.transmat + 1e-8)
            
            omega = torch.zeros(T, self.n_components, device=self.device)
            omega[0] = torch.log(self.startprob + 1e-8) + log_e[0]
            backpointers = torch.zeros(T, self.n_components, dtype=torch.long, device=self.device)
            
            for t in range(1, T):
                step = omega[t-1].unsqueeze(1) + log_transmat
                max_vals, best_prev = torch.max(step, dim=0)
                omega[t] = max_vals + log_e[t]
                backpointers[t] = best_prev
                
            states = torch.zeros(T, dtype=torch.long, device=self.device)
            max_prob, last_state = torch.max(omega[T-1], dim=0)
            states[T-1] = last_state
            
            for t in range(T-2, -1, -1):
                states[t] = backpointers[t+1, states[t+1]]
                
            return max_prob.item(), states.cpu().numpy()
