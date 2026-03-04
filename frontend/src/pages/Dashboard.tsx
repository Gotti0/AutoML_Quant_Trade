import React, { useEffect } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { fetchDashboardData } from '../services/api';
import { AlgorithmCard } from '../components/AlgorithmCard';
import { RegimeWidget } from '../components/RegimeWidget';
import { PipelinePanel } from '../components/PipelinePanel';
import { Activity, LayoutDashboard, RefreshCcw } from 'lucide-react';

export const Dashboard: React.FC = () => {
    const { algorithms, regimeHistory, currentRegime, isLoading, error, setAlgorithms, setRegimeHistory, setLoading, setError } = useDashboardStore();

    const loadData = async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await fetchDashboardData();
            setAlgorithms(data.algorithms);
            setRegimeHistory([data.currentRegime]); // backend API currently sends only the latest regime
        } catch (err: any) {
            setError(err.message || 'Failed to fetch dashboard data');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
        // Optional: Set up polling or WebSocket connection here
        const interval = setInterval(loadData, 30000); // Poll every 30s
        return () => clearInterval(interval);
    }, []);

    const latestRegime = regimeHistory.length > 0 ? regimeHistory[regimeHistory.length - 1] : null;

    return (
        <div className="min-h-screen bg-slate-900 text-slate-200">
            {/* Header */}
            <header className="bg-slate-800 border-b border-slate-700 sticky top-0 z-10 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <LayoutDashboard className="text-indigo-400 w-6 h-6" />
                    <h1 className="text-xl font-bold tracking-tight text-slate-100">AutoML Quant Trade Dashboard</h1>
                </div>
                <div className="flex items-center gap-4">
                    {isLoading && <span className="text-sm text-slate-400 animate-pulse">Syncing...</span>}
                    <button
                        onClick={loadData}
                        disabled={isLoading}
                        className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm font-medium transition-colors disabled:opacity-50"
                    >
                        <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                        Refresh
                    </button>
                </div>
            </header>

            <main className="max-w-7xl mx-auto p-6 space-y-8">
                {error && (
                    <div className="p-4 bg-rose-500/10 border border-rose-500/50 rounded-lg text-rose-400">
                        <p>Error: {error}</p>
                    </div>
                )}

                {/* Pipeline Control Panel */}
                <section>
                    <PipelinePanel />
                </section>

                {/* Top Section: Regime Status & Portfolio Overview */}
                <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2">
                        <RegimeWidget regimeData={latestRegime} />
                    </div>
                    <div className="p-6 bg-slate-800 rounded-xl border border-slate-700 shadow-lg flex flex-col justify-center items-center text-center">
                        <h3 className="text-slate-400 font-medium mb-2">Total Managed Capital</h3>
                        <p className="text-4xl font-extrabold text-white">₩100,000,000</p>
                        <p className="text-sm text-emerald-400 mt-2">+2.4% Today</p>
                    </div>
                </section>

                {/* Bottom Section: Sub-Engine Performances */}
                <section>
                    <div className="flex items-center gap-2 mb-6">
                        <Activity className="text-indigo-400 w-5 h-5" />
                        <h2 className="text-2xl font-bold text-slate-100">Engine Performances</h2>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {algorithms.length > 0 ? (
                            algorithms.map((algo) => (
                                <AlgorithmCard key={algo.id} model={algo} />
                            ))
                        ) : (
                            <div className="col-span-full py-12 text-center text-slate-500">
                                {isLoading ? 'Loading engines...' : 'No algorithms found. Run a backtest first.'}
                            </div>
                        )}
                    </div>
                </section>
            </main>
        </div>
    );
};
