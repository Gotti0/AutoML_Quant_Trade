import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { RefreshCw, AlertTriangle, AlertCircle, Info, Database, Trash2 } from 'lucide-react';
import { clearSystemLogs } from '../services/api';

interface SystemLog {
    id: number;
    timestamp: string;
    level: string;
    source: string;
    message: string;
}

export const SystemLogsPanel: React.FC = () => {
    const [logs, setLogs] = useState<SystemLog[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [clearing, setClearing] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const fetchLogs = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get('http://localhost:8000/api/v1/pipeline/logs?limit=50');
            setLogs(response.data.logs || []);
        } catch (err) {
            let msg = 'Unknown error';
            if (err instanceof Error) msg = err.message;
            setError(`Failed to fetch system logs: ${msg}`);
        } finally {
            setLoading(false);
        }
    };

    const handleClearLogs = async () => {
        if (!window.confirm("Are you sure you want to clear ALL system logs? This action cannot be undone.")) return;

        setClearing(true);
        try {
            await clearSystemLogs();
            await fetchLogs(); // 비운 뒤 즉시 목록 갱신
        } catch (err: any) {
            setError(err.message || 'Failed to clear system logs');
        } finally {
            setClearing(false);
        }
    };

    useEffect(() => {
        fetchLogs();
        // 10초마다 자동 갱신
        const interval = setInterval(fetchLogs, 10000);
        return () => clearInterval(interval);
    }, []);

    const getLevelIcon = (level: string) => {
        switch (level.toUpperCase()) {
            case 'ERROR': return <AlertCircle className="w-4 h-4 text-rose-400" />;
            case 'WARNING': return <AlertTriangle className="w-4 h-4 text-amber-400" />;
            case 'INFO': return <Info className="w-4 h-4 text-sky-400" />;
            default: return <Database className="w-4 h-4 text-slate-400" />;
        }
    };

    const getSourceColor = (source: string) => {
        switch (source) {
            case 'backend': return 'text-purple-400 border-purple-400/30 bg-purple-400/10';
            case 'bridge': return 'text-emerald-400 border-emerald-400/30 bg-emerald-400/10';
            case 'frontend': return 'text-blue-400 border-blue-400/30 bg-blue-400/10';
            default: return 'text-slate-400 border-slate-400/30 bg-slate-400/10';
        }
    };

    return (
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-xl overflow-hidden flex flex-col h-full">
            <div className="p-4 border-b border-slate-700/50 flex justify-between items-center bg-slate-800/80">
                <div className="flex items-center gap-2">
                    <Database className="w-5 h-5 text-indigo-400" />
                    <h2 className="text-lg font-semibold text-slate-100">System Logs (DB)</h2>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={handleClearLogs}
                        disabled={clearing || loading || logs.length === 0}
                        className="flex items-center gap-1.5 px-3 py-1.5 hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 rounded-lg transition-colors disabled:opacity-50 text-sm font-medium border border-transparent hover:border-rose-500/30"
                        title="Clear all logs"
                    >
                        <Trash2 className={`w-4 h-4 ${clearing ? 'animate-bounce' : ''}`} />
                        <span className="hidden sm:inline">Clear</span>
                    </button>
                    <div className="w-px h-6 bg-slate-700 mx-1"></div>
                    <button
                        onClick={fetchLogs}
                        disabled={loading || clearing}
                        className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-400 hover:text-white disabled:opacity-50"
                        title="Refresh logs"
                    >
                        <RefreshCw className={`w-4 h-4 ${(loading && !clearing) ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 bg-slate-900/50">
                {error ? (
                    <div className="text-rose-400 text-center py-6 flex flex-col items-center gap-2">
                        <AlertCircle className="w-8 h-8" />
                        <p>{error}</p>
                    </div>
                ) : logs.length === 0 ? (
                    <div className="text-slate-500 text-center py-8">
                        No logs recorded yet or waiting for DB initialization.
                    </div>
                ) : (
                    <div className="space-y-2">
                        {logs.map((log) => (
                            <div key={log.id} className="flex gap-3 text-sm font-mono items-start p-2 rounded hover:bg-slate-800/50 transition-colors border-b border-slate-800/50 last:border-0">
                                <div className="flex-shrink-0 pt-0.5" title={log.level}>
                                    {getLevelIcon(log.level)}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="text-slate-500 text-xs">
                                            {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                        </span>
                                        <span className={`text-[10px] px-1.5 py-0.5 rounded border ${getSourceColor(log.source)}`}>
                                            {log.source.toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-slate-300 break-words whitespace-pre-wrap">{log.message}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};
