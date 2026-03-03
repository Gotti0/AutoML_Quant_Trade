"""
AutoML_Quant_Trade - 트리플 배리어 기반 메타 레이블링 모듈

미래참조 편향 없이 T 시점의 특성(Feature)에 대응되는 레이블을 T+1~T+horizon 기간의
목표수익률(Take-profit) 달성, 손절매(Stop-loss) 달성, 또는 시간초과(Vertical barrier)로 분류합니다.
"""

import numpy as np
import pandas as pd

def extract_triple_barrier_labels(
    df: pd.DataFrame, 
    vol_window: int = 20, 
    horizon: int = 5, 
    tp_mult: float = 2.0, 
    sl_mult: float = 1.0
) -> pd.DataFrame:
    """
    Pandas 및 Numpy 벡터화 연산을 통해 트리플 배리어 레이블링을 수행합니다.
    
    Args:
        df: 'Close' 컬럼이 포함된 가격 데이터 DataFrame
        vol_window: 변동성 롤링 계산 윈도우 (기본: 20)
        horizon: 탐색할 미래 기간 (수직 장벽, Vertical Barrier) (기본: 5)
        tp_mult: Take-profit 임계값 결정 배수 (upper barrier) (기본: 2.0 sigma)
        sl_mult: Stop-loss 임계값 결정 배수 (lower barrier) (기본: 1.0 sigma)
        
    Returns:
        pd.DataFrame: 원본 df에 다음과 같은 컬럼이 추가됨:
            - 'volatility': 롤링 변동성
            - 'target_label': 1(상승돌파), -1(하락돌파), 0(수직만료)
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")
        
    out = df.copy()
    
    # 1. 롤링 변동성 (표준편차) 산출
    # 일일 수익률 기반
    returns = out['Close'].pct_change()
    out['volatility'] = returns.rolling(window=vol_window).std()
    
    # 2. 결과 담을 배열 초기화 (기본값 0: 수직 장벽 만료)
    n = len(out)
    labels = np.zeros(n, dtype=int)
    
    # numpy array 추출로 for-loop 속도 향상
    close_arr = out['Close'].values
    vol_arr = out['volatility'].values
    
    # 마지막 horizon 기간은 미래 데이터가 부족하므로 제외시키기 위해 n - horizon 까지만 순회
    for i in range(vol_window, n - horizon):
        vol_t = vol_arr[i]
        if pd.isna(vol_t) or vol_t == 0:
            continue
            
        p_t = close_arr[i]
        
        # 장벽 임계 가격
        upper_barrier = p_t * (1.0 + (tp_mult * vol_t))
        lower_barrier = p_t * (1.0 - (sl_mult * vol_t))
        
        # T+1 부터 T+horizon 까지 탐색
        hit = 0
        for j in range(1, horizon + 1):
            future_p = close_arr[i + j]
            if future_p >= upper_barrier:
                hit = 1
                break
            elif future_p <= lower_barrier:
                hit = -1
                break
                
        labels[i] = hit
        
    out['target_label'] = labels
    
    # 미래 데이터 부족으로 계산 불가능한 마지막 구간은 NaN 처리 후 드랍 유도
    # (실제 학습 시에는 dropna() 처리 필요)
    out.loc[out.index[-horizon:], 'target_label'] = np.nan
    out.loc[out.index[:vol_window], 'target_label'] = np.nan
    
    return out
