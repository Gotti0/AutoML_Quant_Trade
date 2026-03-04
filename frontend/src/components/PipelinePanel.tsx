import React, { useEffect, useRef } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';
import { startPipeline, subscribePipelineLogs, fetchPipelineCommands } from '../services/api';
import { Terminal, Play, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

const MAX_LOGS = 500;

const LOG_COLORS: Record<string, string> = {
    INFO: 'text-emerald-400',
    WARNING: 'text-amber-400',
    ERROR: 'text-rose-400',
    DEBUG: 'text-slate-500',
};

export const PipelinePanel: React.FC = () => {
    const {
        isRunning, currentCommand, taskId, logs, error, commands,
        setRunning, setCurrentCommand, setTaskId, addLog, clearLogs, setError, setCommands, reset,
    } = usePipelineStore();

    const [selectedCommand, setSelectedCommand] = React.useState('collect-macro');
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const unsubRef = useRef<(() => void) | null>(null);

    // 명령어 목록 로드
    useEffect(() => {
        fetchPipelineCommands()
            .then(setCommands)
            .catch(() => {
                // 백엔드 미연결 시 기본값
                setCommands([
                    { command: 'collect-insert', description: '신규 종목 전체 수집' },
                    { command: 'collect-update', description: '기존 종목 증분 수집' },
                    { command: 'collect-overseas', description: '해외 자산만 수집' },
                    { command: 'collect-macro', description: '거시지표만 수집' },
                    { command: 'train-regime', description: '국면 모델 학습 (Phase 2)' },
                    { command: 'backtest', description: '백테스팅 실행 (Phase 3)' },
                ]);
            });
        return () => {
            if (unsubRef.current) unsubRef.current();
        };
    }, []);

    // 로그 자동 스크롤
    useEffect(() => {
        if (scrollContainerRef.current) {
            scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const handleStart = async () => {
        try {
            clearLogs();
            setError(null);
            setRunning(true);
            setCurrentCommand(selectedCommand);

            const result = await startPipeline(selectedCommand);
            setTaskId(result.taskId);

            // SSE 구독 시작
            const unsub = subscribePipelineLogs(
                result.taskId,
                (log) => addLog(log),
                (completeResult) => {
                    setRunning(false);
                    if (completeResult.status === 'failed') {
                        setError(completeResult.error || 'Pipeline failed');
                    }
                },
            );
            unsubRef.current = unsub;

        } catch (err: any) {
            setRunning(false);
            setError(err.response?.data?.detail || err.message || 'Failed to start pipeline');
        }
    };

    const statusIcon = isRunning
        ? <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
        : error
            ? <AlertCircle className="w-4 h-4 text-rose-400" />
            : logs.length > 0
                ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                : null;

    return (
        <div className="bg-slate-800 rounded-xl border border-slate-700 shadow-lg overflow-hidden">
            {/* Header */}
            <div className="px-5 py-4 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-indigo-400" />
                    <h3 className="font-bold text-slate-100">Pipeline Control</h3>
                    {statusIcon}
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={selectedCommand}
                        onChange={(e) => setSelectedCommand(e.target.value)}
                        disabled={isRunning}
                        className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                        {(commands.length > 0 ? commands : [
                            { command: 'collect-insert', description: '신규 종목 전체 수집' },
                            { command: 'collect-update', description: '기존 종목 증분 수집' },
                            { command: 'collect-overseas', description: '해외 자산만 수집' },
                            { command: 'collect-macro', description: '거시지표만 수집' },
                            { command: 'train-regime', description: '국면 모델 학습 (Phase 2)' },
                            { command: 'backtest', description: '백테스팅 실행 (Phase 3)' },
                        ]).map((cmd) => (
                            <option key={cmd.command} value={cmd.command}>
                                {cmd.command} — {cmd.description}
                            </option>
                        ))}
                    </select>
                    <button
                        onClick={handleStart}
                        disabled={isRunning}
                        className="flex items-center gap-1.5 px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isRunning ? (
                            <><Loader2 className="w-4 h-4 animate-spin" /> Running...</>
                        ) : (
                            <><Play className="w-4 h-4" /> Run</>
                        )}
                    </button>
                </div>
            </div>

            {/* Error Banner */}
            {error && (
                <div className="px-5 py-2 bg-rose-500/10 border-b border-rose-500/30 text-rose-400 text-sm flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span>{error}</span>
                </div>
            )}

            {/* Log Terminal */}
            <div
                ref={scrollContainerRef}
                className="px-5 py-4 font-mono text-sm leading-6 overflow-y-auto bg-slate-900/50"
                style={{ height: '600px', contain: 'strict' }}
            >
                {logs.length === 0 ? (
                    <div className="text-slate-600 text-center py-8">
                        Select a pipeline command and click Run to start
                    </div>
                ) : (
                    logs.map((log, idx) => (
                        <div key={idx} className={`${LOG_COLORS[log.level] || 'text-slate-400'} whitespace-pre-wrap break-all`}>
                            <span className="text-slate-600 select-none">[{log.level.padEnd(7)}] </span>
                            {log.message}
                        </div>
                    ))
                )}
            </div>

            {/* Footer */}
            {logs.length > 0 && (
                <div className="px-5 py-2 border-t border-slate-700 text-xs text-slate-500 flex items-center justify-between">
                    <span>표시 중: {logs.length}/{MAX_LOGS} (최근 로그)</span>
                    {!isRunning && (
                        <button
                            onClick={clearLogs}
                            className="text-slate-400 hover:text-slate-200 transition-colors"
                        >
                            Clear
                        </button>
                    )}
                </div>
            )}
        </div>
    );
};
