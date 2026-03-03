"""
AutoML_Quant_Trade - 이벤트 기반 백테스팅 메인 루프

시간순 이벤트 큐를 순회하며 전략 시그널 → 주문 → 체결을 처리.

■ 미래참조 편향 방지:
  1. MarketEvent는 시간순으로만 처리 (PriorityQueue)
  2. 전략은 current_time 이전 데이터만 접근 가능
  3. 시그널 발생 시점 t의 체결은 t+1 시가 기준
"""
import logging
from queue import PriorityQueue
from typing import Dict, List, Optional

import pandas as pd

from backend.engine.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from backend.engine.ledger import MasterLedger
from backend.engine.transaction_model import TransactionModel

logger = logging.getLogger(__name__)


class BacktestEventLoop:
    """이벤트 기반 백테스팅 엔진"""

    def __init__(self,
                 ledger: MasterLedger,
                 transaction_model: TransactionModel = None):
        self.ledger = ledger
        self.tx_model = transaction_model or TransactionModel()
        self.strategies: Dict[str, list] = {}
        self.current_prices: Dict[str, float] = {}
        self._pending_signals: List[tuple] = []

    def register_strategy(self, engine_name: str, strategy):
        """전략을 특정 서브 엔진에 등록."""
        if engine_name not in self.strategies:
            self.strategies[engine_name] = []
        self.strategies[engine_name].append(strategy)
        logger.info(f"Strategy '{strategy.__class__.__name__}' registered to '{engine_name}'")

    def run(self, market_data: Dict[str, pd.DataFrame],
            timeframe: str = "daily") -> pd.DataFrame:
        """
        백테스팅 실행.

        Parameters:
            market_data: {ticker: DataFrame[date, open, high, low, close, volume]}
            timeframe: "daily" | "minute"
        Returns:
            에퀴티 커브 DataFrame
        """
        pq = self._build_event_queue(market_data, timeframe)
        total_events = pq.qsize()
        logger.info(f"Backtesting started: {total_events} events, "
                     f"{len(market_data)} tickers")

        event_count = 0

        while not pq.empty():
            _, _, event = pq.get()
            event_count += 1

            # 현재 가격 갱신
            self.current_prices[event.ticker] = event.close

            # 대기 시그널 체결 (이전 봉 시그널 → 현재 봉 시가)
            self._process_pending_signals(event)

            # 전략에 MarketEvent 전달
            for engine_name, strategy_list in self.strategies.items():
                for strategy in strategy_list:
                    signal = strategy.on_market_data(event)
                    if signal is not None:
                        self._pending_signals.append((engine_name, signal))

            # 에퀴티 스냅샷
            self.ledger.record_equity(event.timestamp, self.current_prices)

            if event_count % 1000 == 0:
                logger.info(f"  Progress: {event_count}/{total_events}")

        # 결과
        metrics = self.ledger.get_performance_metrics()
        logger.info(f"Backtesting complete: {event_count} events processed")
        if metrics:
            logger.info(
                f"  Return: {metrics.get('total_return', 0):.2%}, "
                f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"  MDD: {metrics.get('max_drawdown', 0):.2%}"
            )

        return self.ledger.get_equity_curve()

    def _process_pending_signals(self, current_event: MarketEvent):
        """
        대기 중인 시그널을 현재 봉의 시가로 체결.

        ■ 미래참조 편향 방지:
          - 시그널은 이전 봉(t)에서 발생
          - 체결은 현재 봉(t+1)의 시가 기준
        """
        remaining = []

        for engine_name, signal in self._pending_signals:
            if signal.ticker == current_event.ticker:
                account = self.ledger.sub_accounts.get(engine_name)
                if account is None:
                    continue

                order = self._signal_to_order(signal, account)
                if order is not None:
                    fill = self.tx_model.simulate_fill(order, current_event)
                    self.ledger.process_fill(engine_name, fill)
            else:
                remaining.append((engine_name, signal))

        self._pending_signals = remaining

    def _signal_to_order(self, signal: SignalEvent,
                         account) -> Optional[OrderEvent]:
        """시그널을 주문으로 변환 (고정 비율 포지션 사이징)."""
        price = self.current_prices.get(signal.ticker, 0)
        if price <= 0:
            return None

        if signal.direction == "BUY":
            available = account.cash * 0.1 * signal.strength
            qty = int(available / price)
            if qty <= 0:
                return None
            return OrderEvent(
                timestamp=signal.timestamp,
                ticker=signal.ticker,
                qty=qty,
                side="BUY",
            )
        elif signal.direction == "SELL":
            held = account.positions.get(signal.ticker, 0)
            sell_qty = max(1, int(held * signal.strength))
            if sell_qty <= 0 or held <= 0:
                return None
            return OrderEvent(
                timestamp=signal.timestamp,
                ticker=signal.ticker,
                qty=sell_qty,
                side="SELL",
            )
        elif signal.direction == "TARGET":
            equity = account.get_equity(self.current_prices)
            target_value = equity * signal.strength
            
            current_qty = account.positions.get(signal.ticker, 0)
            current_value = current_qty * price
            
            diff_value = target_value - current_value
            
            # 3% 이상(임의 임계값) 차이가 날 때만 불필요한 거래 방지
            # 단, Strategy 쪽에서 threshold를 엄격하게 다루지 않으므로 여기서 대략 1% 차이 이상 반영하도록 필터링
            if abs(diff_value) / equity < 0.01 if equity > 0 else True:
                # 초기 자본이 충분치 않아 비율이 미미한 경우엔 생략
                if equity > 0 and abs(diff_value) / equity < 0.01:
                    return None
                    
            qty_diff = int(diff_value / price)
            
            if qty_diff > 0:
                return OrderEvent(timestamp=signal.timestamp, ticker=signal.ticker, qty=qty_diff, side="BUY")
            elif qty_diff < 0:
                return OrderEvent(timestamp=signal.timestamp, ticker=signal.ticker, qty=abs(qty_diff), side="SELL")

        return None

    def _build_event_queue(self, market_data: Dict[str, pd.DataFrame],
                           timeframe: str) -> PriorityQueue:
        """가격 DataFrame을 시간순 MarketEvent 큐로 변환."""
        pq = PriorityQueue()
        event_id = 0

        for ticker, df in market_data.items():
            for _, row in df.iterrows():
                event = MarketEvent(
                    timestamp=int(row["date"]),
                    ticker=ticker,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    timeframe=timeframe,
                )
                pq.put((event.timestamp, event_id, event))
                event_id += 1

        return pq
