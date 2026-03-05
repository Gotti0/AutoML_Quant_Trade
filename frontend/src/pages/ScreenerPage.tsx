import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, RefreshCcw, Search, Filter } from 'lucide-react';
import { fetchScreenerData } from '../services/api';
import { ScreenerData, ScreenerStock } from '../types';
import { ScreenerTable } from '../components/ScreenerTable';
import { StockDetailCard } from '../components/StockDetailCard';

const REGIME_BADGE: Record<string, string> = {
    Bull: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    Bear: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    Crash: 'text-rose-400 bg-rose-400/10 border-rose-400/30',
    Unknown: 'text-slate-400 bg-slate-400/10 border-slate-400/30',
};

const ALL = '__all__';

export const ScreenerPage: React.FC = () => {
    const [data, setData] = useState<ScreenerData | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selected, setSelected] = useState<ScreenerStock | null>(null);

    // 필터 상태
    const [tierFilter, setTierFilter] = useState<string>(ALL);
    const [clusterFilter, setClusterFilter] = useState<string>(ALL);
    const [query, setQuery] = useState('');

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);
            const result = await fetchScreenerData();
            setData(result);
        } catch (err: any) {
            setError(err.message ?? 'Failed to fetch screener data');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const clusterIds = useMemo(() => {
        if (!data) return [];
        const ids = [...new Set(data.stocks.map((s) => s.clusterId))].sort((a, b) => a - b);
        return ids;
    }, [data]);

    const filtered = useMemo(() => {
        if (!data) return [];
        return data.stocks.filter((s) => {
            if (tierFilter !== ALL && s.tier !== tierFilter) return false;
            if (clusterFilter !== ALL && String(s.clusterId) !== clusterFilter) return false;
            if (query) {
                const q = query.toLowerCase();
                if (!s.ticker.toLowerCase().includes(q) && !s.name.toLowerCase().includes(q)) return false;
            }
            return true;
        });
    }, [data, tierFilter, clusterFilter, query]);

    const regime = data?.regime ?? 'Unknown';
    const probs = data?.regimeProbs ?? { Bull: 0, Bear: 0, Crash: 0 };

    return (
        <div className="min-h-screen bg-slate-900 text-slate-200">
            {/* Page header */}
            <header className="bg-slate-800 border-b border-slate-700 sticky top-10 z-10 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Activity className="text-indigo-400 w-6 h-6" />
                    <h1 className="text-xl font-bold tracking-tight text-slate-100">Stock Screener</h1>
                    {data?.timestamp && (
                        <span className="text-xs text-slate-500 ml-2">기준일: {data.timestamp}</span>
                    )}
                </div>
                <button
                    onClick={load}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm font-medium transition-colors disabled:opacity-50"
                >
                    <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </header>

            <main className="max-w-7xl mx-auto p-6 space-y-6">
                {error && (
                    <div className="p-4 bg-rose-500/10 border border-rose-500/50 rounded-lg text-rose-400 text-sm">
                        {error}
                    </div>
                )}

                {/* 국면 요약 카드 */}
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                    <div className="lg:col-span-1 bg-slate-800 rounded-xl border border-slate-700 p-5 flex flex-col justify-center items-center text-center gap-2">
                        <p className="text-xs text-slate-400 uppercase tracking-wide">현재 국면</p>
                        <span className={`px-3 py-1 text-lg font-bold rounded border ${REGIME_BADGE[regime] ?? REGIME_BADGE['Unknown']}`}>
                            {regime}
                        </span>
                    </div>
                    {(['Bull', 'Bear', 'Crash'] as const).map((r) => (
                        <div key={r} className="bg-slate-800 rounded-xl border border-slate-700 p-5 flex flex-col justify-center items-center text-center gap-1">
                            <p className="text-xs text-slate-400 uppercase tracking-wide">{r}</p>
                            <p className={`text-2xl font-bold ${r === 'Bull' ? 'text-emerald-400' : r === 'Bear' ? 'text-amber-400' : 'text-rose-400'}`}>
                                {((probs[r] ?? 0) * 100).toFixed(1)}%
                            </p>
                            {/* Progress bar */}
                            <div className="w-full bg-slate-700 rounded-full h-1.5 mt-1">
                                <div
                                    className={`h-1.5 rounded-full ${r === 'Bull' ? 'bg-emerald-400' : r === 'Bear' ? 'bg-amber-400' : 'bg-rose-400'}`}
                                    style={{ width: `${(probs[r] ?? 0) * 100}%` }}
                                />
                            </div>
                        </div>
                    ))}
                </div>

                {/* 필터 바 */}
                <div className="flex flex-wrap items-center gap-3">
                    {/* 검색 */}
                    <div className="relative flex-1 min-w-[200px]">
                        <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                        <input
                            type="text"
                            placeholder="종목코드 / 종목명 검색..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                    </div>

                    {/* 티어 필터 */}
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-500" />
                        <span className="text-xs text-slate-500">티어:</span>
                        {[ALL, 'A', 'B', 'C', 'D', 'F'].map((t) => (
                            <button
                                key={t}
                                onClick={() => setTierFilter(t)}
                                className={`px-2.5 py-1 text-xs font-semibold rounded transition-colors border ${
                                    tierFilter === t
                                        ? 'bg-indigo-500 border-indigo-500 text-white'
                                        : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500'
                                }`}
                            >
                                {t === ALL ? '전체' : t}
                            </button>
                        ))}
                    </div>

                    {/* 군집 필터 */}
                    {clusterIds.length > 0 && (
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-500">군집:</span>
                            <select
                                value={clusterFilter}
                                onChange={(e) => setClusterFilter(e.target.value)}
                                className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
                            >
                                <option value={ALL}>전체</option>
                                {clusterIds.map((id) => (
                                    <option key={id} value={String(id)}>C{id}</option>
                                ))}
                            </select>
                        </div>
                    )}

                    {/* 필터 리셋 */}
                    {(tierFilter !== ALL || clusterFilter !== ALL || query) && (
                        <button
                            onClick={() => { setTierFilter(ALL); setClusterFilter(ALL); setQuery(''); }}
                            className="text-xs text-slate-500 hover:text-slate-300 underline transition-colors"
                        >
                            필터 초기화
                        </button>
                    )}
                </div>

                {/* 테이블 */}
                {isLoading && !data ? (
                    <div className="text-center py-12 text-slate-500 text-sm animate-pulse">스크리너 데이터 로딩 중...</div>
                ) : (
                    <ScreenerTable stocks={filtered} onSelect={setSelected} />
                )}
            </main>

            {/* 종목 상세 모달 */}
            {selected && (
                <StockDetailCard stock={selected} onClose={() => setSelected(null)} />
            )}
        </div>
    );
};
