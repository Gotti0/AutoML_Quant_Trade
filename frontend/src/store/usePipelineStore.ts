import { create } from 'zustand';
import { PipelineLog, PipelineCommand } from '../types';

const MAX_LOGS = 500; // 브라우저 메모리 보호 — ring buffer

interface PipelineState {
    isRunning: boolean;
    currentCommand: string | null;
    taskId: string | null;
    logs: PipelineLog[];
    error: string | null;
    commands: PipelineCommand[];

    setRunning: (running: boolean) => void;
    setCurrentCommand: (command: string | null) => void;
    setTaskId: (taskId: string | null) => void;
    addLog: (log: PipelineLog) => void;
    clearLogs: () => void;
    setError: (error: string | null) => void;
    setCommands: (commands: PipelineCommand[]) => void;
    reset: () => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
    isRunning: false,
    currentCommand: null,
    taskId: null,
    logs: [],
    error: null,
    commands: [],

    setRunning: (running) => set({ isRunning: running }),
    setCurrentCommand: (command) => set({ currentCommand: command }),
    setTaskId: (taskId) => set({ taskId }),
    addLog: (log) => set((state) => ({
        logs: [...state.logs, log].slice(-MAX_LOGS), // OOM 방지: 최대 500건 유지
    })),
    clearLogs: () => set({ logs: [] }),
    setError: (error) => set({ error }),
    setCommands: (commands) => set({ commands }),
    reset: () => set({
        isRunning: false,
        currentCommand: null,
        taskId: null,
        logs: [],
        error: null,
    }),
}));
