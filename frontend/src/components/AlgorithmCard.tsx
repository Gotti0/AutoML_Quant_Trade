import React from 'react';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { AlgorithmModel } from '../types';

interface Props {
    model: AlgorithmModel;
}

export const AlgorithmCard: React.FC<Props> = ({ model }) => {
    const isPositive = model.metrics.cumulativeReturn > 0;

    return (
        <div className="flex flex-col p-5 bg-slate-800 rounded-xl shadow-lg border border-slate-700 hover:border-slate-500 transition-colors">
            <div className="flex justify-between items-center border-b border-slate-600 pb-2">
                <h3 className="text-xl font-bold text-slate-100">{model.rank}. {model.name}</h3>
                <span className="px-2 py-1 text-xs font-semibold rounded bg-indigo-900 text-indigo-200">
                    {model.timeframe}
                </span>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4 mt-4 text-center">
                <div className="bg-slate-700/50 p-3 rounded-lg">
                    <p className="text-xs text-slate-400 mb-1">Return</p>
                    <p className={`text-lg font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {model.metrics.cumulativeReturn.toFixed(2)}%
                    </p>
                </div>
                <div className="bg-slate-700/50 p-3 rounded-lg">
                    <p className="text-xs text-slate-400 mb-1">MDD</p>
                    <p className="text-lg font-bold text-rose-400">
                        {model.metrics.maxDrawdown.toFixed(2)}%
                    </p>
                </div>
                <div className="bg-slate-700/50 p-3 rounded-lg">
                    <p className="text-xs text-slate-400 mb-1">Sharpe</p>
                    <p className="text-lg font-bold text-sky-400">
                        {model.metrics.sharpeRatio.toFixed(2)}
                    </p>
                </div>
            </div>

            {/* Equity Curve */}
            <div className="h-32 mt-5">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={model.equityCurve}>
                        <defs>
                            <linearGradient id={`colorEquity-${model.id}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={isPositive ? "#10b981" : "#f43f5e"} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={isPositive ? "#10b981" : "#f43f5e"} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '0.5rem', color: '#f1f5f9' }}
                            itemStyle={{ color: '#f1f5f9' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="equity"
                            stroke={isPositive ? "#10b981" : "#f43f5e"}
                            fillOpacity={1}
                            fill={`url(#colorEquity-${model.id})`}
                            strokeWidth={2}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
