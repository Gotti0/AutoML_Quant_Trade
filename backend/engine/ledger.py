"""
AutoML_Quant_Trade - 마스터 원장 + 서브 엔진 계정

계층적 자본 관리:
  MasterLedger
    ├── SubEngineAccount("MidFreq")
    ├── SubEngineAccount("Swing")
    ├── SubEngineAccount("MidShort")
    └── SubEngineAccount("Long_Safe")

각 서브 계정은 독립적 포지션·현금·에퀴티를 추적하며,
마스터 원장이 전체 합산 NAV를 관리.
"""
import logging
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd

from backend.engine.events import FillEvent

logger = logging.getLogger(__name__)


class SubEngineAccount:
    """서브 엔진 독립 계정 — 포지션·현금·에퀴티 관리"""

    def __init__(self, name: str, initial_cash: float):
        self.name = name
        self.cash = initial_cash
        self.positions: Dict[str, int] = defaultdict(int)    # ticker → 수량
        self.avg_cost: Dict[str, float] = defaultdict(float)  # ticker → 평균 단가
        self.equity_history: List[dict] = []
        self.trade_log: List[dict] = []

    def process_fill(self, fill: FillEvent):
        """
        체결 이벤트를 반영하여 포지션·현금 갱신.
        부분 체결을 지원하여 캐시가 마이너스가 되는 현상을 방지함.

        Parameters:
            fill: FillEvent (qty 양수=매수, 음수=매도)
        """
        ticker = fill.ticker
        prev_qty = self.positions[ticker]
        prev_cost = self.avg_cost[ticker]

        if fill.qty > 0:
            # 매수: 현금 감소, 포지션 증가
            # 1. 시도할 총 비용 연산
            attempt_cost = fill.qty * fill.execution_price + fill.fee
            
            # 2. 캐시 부족 시 안전장치 (부분 체결)
            if attempt_cost > self.cash:
                # 보수적으로 수수료 무시하고 단가로 몇 주 살 수 있는지 체크
                max_affordable_qty = int(self.cash / fill.execution_price)
                if max_affordable_qty <= 0:
                    logger.warning(f"[{self.name}] 체결 거부: {ticker} 매수 대금 부족 (필요: {attempt_cost:,.0f}, 보유: {self.cash:,.0f})")
                    return # 캐시가 아예 부족하면 체결 전체 취소
                
                # 수량 강제 조정 및 필요 비용 재산출 (비율에 맞춰 수수료 단순 삭감)
                fee_ratio = fill.fee / attempt_cost
                new_qty = max_affordable_qty
                new_fee = fill.fee * (new_qty / fill.qty)
                cost = new_qty * fill.execution_price + new_fee
                
                # 그래도 오차로 초과하면 추가 삭감
                while cost > self.cash and new_qty > 0:
                    new_qty -= 1
                    new_fee = fill.fee * (new_qty / fill.qty)
                    cost = new_qty * fill.execution_price + new_fee
                
                if new_qty <= 0:
                    return
                    
                logger.warning(f"[{self.name}] 부분 체결: {ticker} 수량 조정 ({fill.qty} -> {new_qty}) - 보유 원금 한계")
                fill.qty = new_qty
                fill.fee = new_fee
            else:
                cost = attempt_cost
            
            self.cash -= cost

            # 평균 단가 갱신 (가중 평균)
            total_qty = prev_qty + fill.qty
            if total_qty > 0:
                self.avg_cost[ticker] = (
                    (prev_qty * prev_cost + fill.qty * fill.execution_price)
                    / total_qty
                )
            self.positions[ticker] = total_qty

        else:
            # 매도: 현금 증가, 포지션 감소
            sell_qty = abs(fill.qty)
            proceeds = sell_qty * fill.execution_price - fill.fee
            self.cash += proceeds
            self.positions[ticker] = prev_qty - sell_qty

            # 포지션이 0이면 평균 단가 초기화
            if self.positions[ticker] <= 0:
                self.positions[ticker] = 0
                self.avg_cost[ticker] = 0.0

        # 거래 로그 기록
        self.trade_log.append({
            "timestamp": fill.timestamp,
            "ticker": ticker,
            "qty": fill.qty,
            "price": fill.execution_price,
            "fee": fill.fee,
            "slippage": fill.slippage,
            "cash_after": self.cash,
        })

    def get_equity(self, market_prices: Dict[str, float]) -> float:
        """
        현재 총 자산 가치(NAV) 계산.

        Parameters:
            market_prices: {ticker: 현재가} (평가용)
        """
        position_value = sum(
            qty * market_prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
            if qty > 0
        )
        return self.cash + position_value

    def record_equity(self, timestamp: int, market_prices: Dict[str, float]):
        """에퀴티 스냅샷 기록."""
        equity = self.get_equity(market_prices)
        self.equity_history.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": self.cash,
            "position_value": equity - self.cash,
        })

    def get_equity_curve(self) -> pd.DataFrame:
        """에퀴티 커브 DataFrame 반환."""
        if not self.equity_history:
            return pd.DataFrame(columns=["timestamp", "equity", "cash", "position_value"])
        return pd.DataFrame(self.equity_history)

    def get_trade_log_df(self) -> pd.DataFrame:
        """거래 로그 DataFrame 반환."""
        if not self.trade_log:
            return pd.DataFrame(columns=["timestamp", "ticker", "qty", "price", "fee"])
        return pd.DataFrame(self.trade_log)

    def get_performance_metrics(self) -> Dict:
        """서브 계정에 대한 핵심 성과 지표 계산."""
        curve = self.get_equity_curve()
        if curve.empty:
            return {}

        equity = curve["equity"]
        returns = equity.pct_change().dropna()

        # 총 수익률
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if equity.iloc[0] != 0 else 0

        # 연간 수익률 (일봉 기준 252일 가정)
        n_days = len(equity)
        try:
            # total_return이 -1 이하가 되면 (1 + total_return)이 음수가 되어
            # 거듭제곱 시 복소수(NaN) 발생 위험이 있으므로 하한(0.0)을 적용.
            annual_return = (max(1 + total_return, 0.0)) ** (252 / max(n_days, 1)) - 1
        except Exception:
            annual_return = 0.0

        # 변동성 (연율화)
        annual_vol = returns.std() * (252 ** 0.5) if len(returns) > 1 else 0

        # 샤프 비율 (무위험 이자율 3% 가정)
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0

        # MDD
        cummax = equity.cummax()
        drawdowns = (equity - cummax) / cummax
        mdd = drawdowns.min() if not pd.isna(drawdowns.min()) else 0.0

        calmar = annual_return / abs(mdd) if mdd != 0 else 0

        # 단순 승률 추정 
        win_rate = 0.5 # Phase 4에서 상세 구현

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "calmar_ratio": float(calmar),
            "total_trades": len(self.trade_log),
            "win_rate": win_rate,
            "final_equity": float(equity.iloc[-1]),
        }

    @property
    def total_positions(self) -> int:
        """보유 종목 수."""
        return sum(1 for q in self.positions.values() if q > 0)


class MasterLedger:
    """마스터 원장 — 서브 엔진 계정들의 통합 관리"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.sub_accounts: Dict[str, SubEngineAccount] = {}
        self.equity_history: List[dict] = []

    def create_sub_account(self, name: str, allocation_ratio: float) -> SubEngineAccount:
        """
        서브 엔진 계정 생성.

        Parameters:
            name: 계정명 (예: "MidFreq", "Swing")
            allocation_ratio: 초기 자본 비율 (0~1)
        Returns:
            생성된 SubEngineAccount
        """
        initial_cash = self.initial_capital * allocation_ratio
        account = SubEngineAccount(name, initial_cash)
        self.sub_accounts[name] = account
        logger.info(f"Sub account '{name}' created: {initial_cash:,.0f} KRW ({allocation_ratio:.0%})")
        return account

    def process_fill(self, engine_name: str, fill: FillEvent):
        """특정 서브 계정에 체결 이벤트 반영."""
        if engine_name not in self.sub_accounts:
            raise ValueError(f"Unknown sub account: {engine_name}")
        self.sub_accounts[engine_name].process_fill(fill)

    def record_equity(self, timestamp: int, market_prices: Dict[str, float]):
        """모든 서브 계정의 에퀴티를 기록하고 합산."""
        total_equity = 0.0
        sub_equities = {}

        for name, account in self.sub_accounts.items():
            account.record_equity(timestamp, market_prices)
            eq = account.get_equity(market_prices)
            sub_equities[name] = eq
            total_equity += eq

        self.equity_history.append({
            "timestamp": timestamp,
            "total_equity": total_equity,
            **sub_equities,
        })

    def get_total_equity(self, market_prices: Dict[str, float]) -> float:
        """전체 NAV 합산."""
        return sum(
            acc.get_equity(market_prices)
            for acc in self.sub_accounts.values()
        )

    def get_equity_curve(self) -> pd.DataFrame:
        """전체 에퀴티 커브 반환."""
        if not self.equity_history:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_history)

    def get_subaccount_returns_tensor(self, window: int = 60, engines_order: List[str] = None):
        """
        최근 window일 간의 서브계정 수익률을 형상 (N_days, M_engines)의 PyTorch Tensor로 반환.
        """
        if len(self.equity_history) < 2:
            return None
            
        import torch
        import pandas as pd
        
        df = pd.DataFrame(self.equity_history)
        if engines_order is None:
            engines_order = list(self.sub_accounts.keys())
            
        # N일치 추출 (수익률 계산을 위해 window+1개 추출)
        df_window = df.tail(window + 1).copy()
        
        # 각 엔진별 칼럼 추출
        equities = df_window[engines_order]
        
        # 수익률: pct_change 후 첫 행(NaN) 드롭
        returns = equities.pct_change().dropna()
        
        return torch.tensor(returns.values, dtype=torch.float32)

    def get_all_trades(self) -> pd.DataFrame:
        """모든 서브 계정의 거래 로그 통합."""
        all_trades = []
        for name, account in self.sub_accounts.items():
            trades = account.get_trade_log_df()
            if not trades.empty:
                trades["engine"] = name
                all_trades.append(trades)

        if not all_trades:
            return pd.DataFrame()
        return pd.concat(all_trades, ignore_index=True).sort_values("timestamp")

    def get_performance_metrics(self) -> Dict:
        """핵심 성과 지표 계산."""
        curve = self.get_equity_curve()
        if curve.empty:
            return {}

        equity = curve["total_equity"]
        returns = equity.pct_change().dropna()

        # 총 수익률
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # 연간 수익률 (일봉 기준 252일 가정)
        n_days = len(equity)
        try:
            annual_return = (max(1 + total_return, 0.0)) ** (252 / max(n_days, 1)) - 1
        except Exception:
            annual_return = 0.0

        # 변동성 (연율화)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            annual_vol = returns.std() * (252 ** 0.5) if len(returns) > 1 else 0

        # 샤프 비율 (무위험 이자율 3% 가정)
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0

        # MDD
        cummax = equity.cummax()
        drawdowns = (equity - cummax) / cummax
        mdd = drawdowns.min()

        # 칼마 비율
        calmar = annual_return / abs(mdd) if mdd != 0 else 0

        # 승률
        all_trades = self.get_all_trades()
        if not all_trades.empty:
            # 간단한 승률: 매도 시 이익인 거래 비율
            sells = all_trades[all_trades["qty"] < 0]
            # (단순화: 매도가 > 매수 평균 단가)
            win_rate = 0.0  # Phase 4에서 상세 구현
        else:
            win_rate = 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "total_trades": len(all_trades) if not all_trades.empty else 0,
            "initial_capital": self.initial_capital,
            "final_equity": float(equity.iloc[-1]),
        }
