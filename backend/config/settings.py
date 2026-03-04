"""
AutoML_Quant_Trade - 전역 설정
"""
import os


class Settings:
    """시스템 전역 설정 상수"""

    # ── 경로 ──
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "cache_daishin")
    DB_PATH = os.path.join(DATA_DIR, "quant_data.db")
    MODEL_DIR = os.path.join(DATA_DIR, "models")

    # ── 32비트 브릿지 서버 ──
    BRIDGE_URL = "http://127.0.0.1:8001"

    # ── 거시지표 해외 코드 (CpSvrNew8300) ──
    MACRO_CODES = {
        "다우존스": ".DJI",
        "나스닥": "COMP",
        "S&P500": "SPX",
        "USD/KRW": "DS#USDKRW",
        "WTI원유": "CM@PWTI",
    }

    # ── 자산 유니버스: 국내 + 해외 직접 투자 ──
    ASSET_UNIVERSE = {
        # 국내 주식 (StockChart 조회)
        "Equity_Domestic": {"source": "domestic", "code": "A069500"},       # KOSPI 200 ETF
        # 해외 주식 (CpSvrNew8300 조회) — 직접 투자
        "Equity_US_Tech": {"source": "overseas", "codes": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]},
        "Equity_US_Index": {"source": "overseas", "codes": [".DJI", "COMP", "SPX"]},
        # 안전자산: 채권 (StockChart 조회)
        "FixedIncome": {"source": "domestic", "code": "A304660"},           # KODEX 미국채울트라30년선물(H)
        # 원자재 (CpSvrNew8300 조회)
        "Commodity_Gold": {"source": "overseas", "codes": ["CM@NGLD"]},
        "Commodity_Oil": {"source": "overseas", "codes": ["CM@PWTI"]},
        # 환율 (CpSvrNew8300 조회)
        "FX_USDKRW": {"source": "overseas", "codes": ["DS#USDKRW"]},
    }

    # ── 백테스팅 파라미터 ──
    INITIAL_CAPITAL = 100_000_000   # 초기 자본금 1억원
    COMMISSION_RATE = 0.00015       # 매매 수수료 0.015%
    TAX_RATE = 0.0018               # 매도 거래세 0.18%
    MARKET_IMPACT_GAMMA = 0.1       # 시장 충격 계수 (제곱근 모델)
    BID_ASK_SPREAD = 0.0005         # 평균 호가 스프레드 5bp

    # ── 메타 포트폴리오 엔진 ──
    REGIME_COUNT = 3                # HMM 국면 수 (Bull, Bear, Crash)
    REBALANCE_FREQ = "M"            # 리밸런싱 주기 (M=월간)

    # ── 리스크 관리 ──
    ENGINE_LOSS_LIMITS = {
        "MidFreq": 0.05,            # 중빈도 엔진 MDD 한도 5%
        "Swing": 0.10,              # 스윙 엔진 MDD 한도 10%
        "MidShort": 0.15,           # 중단기 엔진 MDD 한도 15%
        "Long_Safe": 0.20,          # 장기 안전자산 MDD 한도 20%
    }

    # ── 데이터 수집 ──
    CYBOS_THROTTLE_WAIT = 0.25      # API 호출 간격 (초) — 15초당 60건 제한 대응
    DEFAULT_DAILY_COUNT = 500       # 기본 일봉 수집 건수
    DEFAULT_MINUTE_COUNT = 5000     # 기본 분봉 수집 건수

    @classmethod
    def ensure_dirs(cls):
        """필요한 디렉토리가 없으면 생성"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
