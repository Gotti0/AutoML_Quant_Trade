import React from 'react';
import { RegimeProbability } from '../types';
import { Activity } from 'lucide-react';

interface Props {
    regimeData: RegimeProbability | null;
}

export const RegimeWidget: React.FC<Props> = ({ regimeData }) => {
    if (!regimeData) {
        return (
            <div className="p-6 bg-slate-800 rounded-xl border border-slate-700 flex items-center justify-center h-48">
                <span className="text-slate-500">No regime data available</span>
            </div>
        );
    }

    const { Bull, Bear, Crash } = regimeData.probabilities;

    const getRegimeColor = (regime: string) => {
        switch (regime.toLowerCase()) {
            case 'bull': return 'text-emerald-400';
            case 'bear': return 'text-amber-400';
            case 'crash': return 'text-rose-500';
            default: return 'text-slate-300';
        }
    };

    return (
        <div className="p-6 bg-slate-800 rounded-xl border border-slate-700 shadow-lg">
            <div className="flex items-center gap-3 mb-6">
                <Activity className="text-sky-400 w-6 h-6" />
                <h2 className="text-xl font-bold text-slate-100">Market Regime</h2>
            </div>

            <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                <div className="flex-1 text-center md:text-left">
                    <p className="text-sm text-slate-400 mb-1">Current Dominant Regime</p>
                    <p className={`text-4xl font-extrabold tracking-tight ${getRegimeColor(regimeData.dominantRegime)}`}>
                        {regimeData.dominantRegime.toUpperCase()}
                    </p>
                    <p className="text-xs text-slate-500 mt-2">Latest Inference: {regimeData.timestamp}</p>
                </div>

                <div className="flex-1 w-full space-y-4">
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-emerald-400 font-medium">Bull</span>
                            <span className="text-slate-300">{(Bull * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full bg-emerald-400" style={{ width: `${Bull * 100}%` }}></div>
                        </div>
                    </div>
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-amber-400 font-medium">Bear</span>
                            <span className="text-slate-300">{(Bear * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full bg-amber-400" style={{ width: `${Bear * 100}%` }}></div>
                        </div>
                    </div>
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-rose-500 font-medium">Crash</span>
                            <span className="text-slate-300">{(Crash * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full bg-rose-500" style={{ width: `${Crash * 100}%` }}></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
