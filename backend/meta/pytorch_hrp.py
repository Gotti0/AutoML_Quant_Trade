import logging
import torch
import numpy as np
from scipy.cluster.hierarchy import linkage

logger = logging.getLogger(__name__)

class TorchHRPOptimizer:
    """
    고속 연산을 위한 PyTorch 기반 HRP (계층적 리스크 패리티) 최적화기.
    SciPy의 계층적 군집화를 활용하는 하이브리드 파이프라인.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def calculate_hrp_weights(self, returns: torch.Tensor) -> torch.Tensor:
        """
        수익률 텐서로부터 HRP 가중치 산출.
        
        Parameters:
            returns: 형상 (N_days, M_engines)의 2D 텐서 (수익률 시계열)
        
        Returns:
            weights: 형상 (M_engines,)의 1D 텐서 (자산 비중, 합=1.0)
        """
        returns = returns.to(self.device, dtype=torch.float32)
        N, M = returns.shape
        
        if M < 2:
            return torch.ones(M, device=self.device)
            
        if N < 2:
            # 데이터가 부족하면 동일 비중 반환
            return torch.ones(M, device=self.device) / M

        # 1. 상관 행렬 연산
        # torch.corrcoef는 (특성 수, 변수 수) 형태가 필요 (M, N)
        returns_t = returns.t()
        corr = torch.corrcoef(returns_t)
        corr = torch.nan_to_num(corr, 0.0)
        
        # 2. 거리 행렬 연산 (D = sqrt(0.5 * (1 - rho)))
        dist_mat = torch.sqrt(0.5 * torch.clamp(1.0 - corr, min=0.0, max=2.0))
        
        # Condensed Distance Matrix 추출 (SciPy linkage 입력용)
        # 상삼각 행렬 추출 (대각 제외)
        idx = torch.triu_indices(M, M, offset=1)
        condensed_dist = dist_mat[idx[0], idx[1]].cpu().numpy()
        
        # 3. CPU에서 SciPy 계층적 군집화 수행
        if len(condensed_dist) == 0:
            link = np.array([])
        else:
            link = linkage(condensed_dist, method='single')
            
        # 4. 준-대각화 (Quasi-Diagonalization) : 클러스터 트리에 따라 자산을 재배열
        sort_idx = self._get_quasi_diag(link, M)
        
        # 5. 재귀적 이등분 (Recursive Bisection)을 통한 비중 산출
        cov = torch.cov(returns_t)
        cov = torch.nan_to_num(cov, 0.0)
        
        weights = self._get_rec_bipart(cov, sort_idx)
        return weights

    def _get_quasi_diag(self, link: np.ndarray, num_items: int) -> list:
        """군집 트리를 기반으로 리프 노드를 정렬하여 준대각화 인덱스 반환."""
        if len(link) == 0:
            return list(range(num_items))
            
        def get_leaves(node):
            if node < num_items:
                return [int(node)]
            else:
                row = int(node - num_items)
                left = link[row, 0]
                right = link[row, 1]
                return get_leaves(left) + get_leaves(right)
                
        root = 2 * num_items - 2
        return get_leaves(root)

    def _get_cluster_var(self, cov: torch.Tensor, c_items: list) -> torch.Tensor:
        """특정 클러스터의 역분산 포트폴리오 가중치에 따른 분산 계산."""
        c_items_tensor = torch.tensor(c_items, device=self.device)
        # 2D 인덱싱을 위한 그리드 생성
        cov_slice = cov[c_items_tensor[:, None], c_items_tensor]
        
        # 대각 성분 (분산) 의 역수
        inv_diag = 1.0 / torch.diag(cov_slice)
        inv_diag = torch.where(torch.isinf(inv_diag), torch.zeros_like(inv_diag), inv_diag)
        inv_diag = torch.nan_to_num(inv_diag, 0.0)
        
        sum_inv = torch.sum(inv_diag)
        if sum_inv == 0:
            w = torch.ones(len(c_items), device=self.device, dtype=torch.float32) / len(c_items)
        else:
            w = inv_diag / sum_inv
            
        # w^T * Cov * w
        var = torch.matmul(torch.matmul(w, cov_slice), w)
        return var

    def _get_rec_bipart(self, cov: torch.Tensor, sort_idx: list) -> torch.Tensor:
        """재귀적 이등분 방식으로 자본 비중을 하향식으로 분배."""
        w = torch.ones(cov.shape[0], device=self.device, dtype=torch.float32)
        clusters = [sort_idx]
        
        while len(clusters) > 0:
            clusters_next = []
            for c in clusters:
                if len(c) > 1:
                    # 절반으로 나누기
                    half = int(len(c) / 2)
                    c0 = c[:half]
                    c1 = c[half:]
                    
                    # 각 하위 클러스터의 분산 연산
                    var0 = self._get_cluster_var(cov, c0)
                    var1 = self._get_cluster_var(cov, c1)
                    
                    # 분산에 반비례하도록 비중 분배 인자(alpha) 설정
                    if var0 + var1 == 0:
                        alpha = torch.tensor(0.5, device=self.device, dtype=torch.float32)
                    else:
                        alpha = var1 / (var0 + var1)
                    alpha = torch.nan_to_num(alpha, 0.5)
                    
                    # 할당
                    w[c0] *= alpha
                    w[c1] *= (1.0 - alpha)
                    
                    clusters_next.append(c0)
                    clusters_next.append(c1)
                    
            clusters = clusters_next
            
        return w
