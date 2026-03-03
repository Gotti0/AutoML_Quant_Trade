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
