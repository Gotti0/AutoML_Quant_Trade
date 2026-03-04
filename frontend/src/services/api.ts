import axios from 'axios';
import { AlgorithmModel, RegimeProbability, PipelineLog, PipelineCommand, PipelineStatus } from '../types';

// Use an environment variable for the base URL in a real app
const API_BASE_URL = 'http://localhost:8000/api/v1';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface DashboardResponse {
    currentRegime: RegimeProbability;
    algorithms: AlgorithmModel[];
}

export const fetchDashboardData = async (): Promise<DashboardResponse> => {
    try {
        const response = await apiClient.get<DashboardResponse>('/dashboard');
        return response.data;
    } catch (error) {
        let msg = 'Unknown error';
        if (error instanceof Error) msg = error.message;
        console.error('Error fetching dashboard data:', error);
        await sendClientLog('ERROR', `Dashboard fetch failed: ${msg}`);
        throw error;
    }
};

export const sendClientLog = async (level: string, message: string): Promise<void> => {
    // API 연결 오류 등 치명적인 상황일 수 있으므로 실패해도 화면을 멈추지 않도록 catch 추가
    try {
        await apiClient.post('/pipeline/logs', { level, message });
    } catch (e) {
        console.error('Failed to send client log to server:', e);
    }
};

export const fetchRegimeHistory = async (): Promise<RegimeProbability[]> => {
    const response = await apiClient.get('/regime/history');
    return response.data;
};

// ── Pipeline API ──

export const startPipeline = async (command: string): Promise<{ taskId: string; command: string; status: string }> => {
    try {
        const response = await apiClient.post('/pipeline/run', null, { params: { command } });
        await sendClientLog('INFO', `Triggered pipeline command: ${command}`);
        return response.data;
    } catch (error) {
        let msg = 'Unknown error';
        if (error instanceof Error) msg = error.message;
        await sendClientLog('ERROR', `Failed to start pipeline ${command}: ${msg}`);
        throw error;
    }
};

export const fetchPipelineStatus = async (): Promise<PipelineStatus> => {
    try {
        const response = await apiClient.get<PipelineStatus>('/pipeline/status');
        return response.data;
    } catch (error) {
        // Status polling은 빈번하므로 에러를 로그 레벨로 조정 (Warning)
        let msg = 'Unknown error';
        if (error instanceof Error) msg = error.message;
        await sendClientLog('WARNING', `Status polling failed: ${msg}`);
        throw error;
    }
};

export const fetchPipelineCommands = async (): Promise<PipelineCommand[]> => {
    try {
        const response = await apiClient.get<{ commands: PipelineCommand[] }>('/pipeline/commands');
        return response.data.commands;
    } catch (error) {
        let msg = 'Unknown error';
        if (error instanceof Error) msg = error.message;
        await sendClientLog('WARNING', `Failed to fetch commands, using fallback. Error: ${msg}`);
        throw error;
    }
};

export const subscribePipelineLogs = (
    taskId: string,
    onLog: (log: PipelineLog) => void,
    onComplete: (result: { status: string; error: string | null }) => void,
): (() => void) => {
    const es = new EventSource(`${API_BASE_URL}/pipeline/logs/${taskId}`);
    es.onmessage = (e) => onLog(JSON.parse(e.data));
    es.addEventListener('complete', (e: Event) => {
        const me = e as MessageEvent;
        onComplete(JSON.parse(me.data));
        es.close();
    });
    es.onerror = () => {
        es.close(); // 자동 재연결 방지 (OOM 위험 제거)
    };
    return () => es.close();
};

// Add fallback mock data generation for testing UI without backend
export const generateMockData = (): DashboardResponse => {
    const currentDate = new Date().toISOString().split('T')[0];

    const generateCurve = (startPoints: number, trend: number, vol: number) => {
        let current = startPoints;
        return Array.from({ length: 30 }, (_, i) => {
            const date = new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0];
            current = current * (1 + (Math.random() - 0.5) * vol + trend);
            return { date, equity: parseFloat(current.toFixed(2)) };
        });
    };

    return {
        currentRegime: {
            timestamp: currentDate,
            probabilities: { Bull: 0.15, Bear: 0.65, Crash: 0.20 },
            dominantRegime: 'Bear'
        },
        algorithms: [
            {
                id: '1', name: 'MidFreq Scalping', timeframe: 'HFT', rank: 1,
                metrics: { cumulativeReturn: 12.4, maxDrawdown: -4.2, sharpeRatio: 1.8, winRate: 65 },
                equityCurve: generateCurve(100, 0.005, 0.02)
            },
            {
                id: '2', name: 'Swing Mean Reversion', timeframe: 'SWING', rank: 2,
                metrics: { cumulativeReturn: 8.2, maxDrawdown: -6.5, sharpeRatio: 1.4, winRate: 58 },
                equityCurve: generateCurve(100, 0.002, 0.03)
            },
            {
                id: '3', name: 'Mid-term Trend', timeframe: 'MID_TERM', rank: 3,
                metrics: { cumulativeReturn: -2.1, maxDrawdown: -12.4, sharpeRatio: -0.2, winRate: 42 },
                equityCurve: generateCurve(100, -0.001, 0.04)
            },
            {
                id: '4', name: 'Long Safe Allocation', timeframe: 'LONG_TERM', rank: 4,
                metrics: { cumulativeReturn: 4.5, maxDrawdown: -2.1, sharpeRatio: 2.1, winRate: 75 },
                equityCurve: generateCurve(100, 0.001, 0.01)
            }
        ]
    };
};

// Temporary interceptor for mock data if backend is offline
if (import.meta.env.VITE_USE_MOCK_DATA === 'true') {
    apiClient.interceptors.response.use(
        (response) => response,
        (error) => {
            console.warn('API connection failed, using mock data for development.', error.message);
            if (error.config.url.includes('/dashboard')) {
                return Promise.resolve({ data: generateMockData() });
            }
            return Promise.reject(error);
        }
    );
}

