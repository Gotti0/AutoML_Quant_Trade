export interface PerformanceMetrics {
    cumulativeReturn: number;
    maxDrawdown: number;
    sharpeRatio: number;
    winRate: number;
}

export interface EquityDataPoint {
    date: string;
    equity: number;
}

export interface AlgorithmModel {
    id: string;
    name: string;
    timeframe: 'HFT' | 'SWING' | 'MID_TERM' | 'LONG_TERM';
    rank: number;
    metrics: PerformanceMetrics;
    equityCurve: EquityDataPoint[];
}

export interface RegimeProbability {
    timestamp: string;
    probabilities: {
        Bull: number;
        Bear: number;
        Crash: number;
    };
    dominantRegime: string;
}

// ── Screener 타입 ──

export interface ScreenerStock {
    ticker: string;
    name: string;
    clusterId: number;
    techScore: number;
    fundScore: number;
    totalScore: number;
    tier: 'A' | 'B' | 'C' | 'D' | 'F';
    isAnomaly: boolean;
    fundamentals: {
        per: number;
        roe: number;
        dividendYield: number;
        debtRatio: number;
        pbr: number;
        eps: number;
    };
}

export interface ScreenerData {
    timestamp: string;
    regime: string;
    regimeProbs: { Bull: number; Bear: number; Crash: number };
    stocks: ScreenerStock[];
}

// ── Pipeline 타입 ──

export interface PipelineLog {
    level: string;
    message: string;
    timestamp: number;
    logger: string;
}

export interface PipelineStatus {
    status: 'idle' | 'running' | 'completed' | 'failed';
    taskId: string | null;
    command: string | null;
    startedAt: string | null;
    error: string | null;
}

export interface PipelineCommand {
    command: string;
    description: string;
}

