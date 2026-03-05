import React from 'react';
import { X, AlertTriangle } from 'lucide-react';
import {
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    Radar,
    ResponsiveContainer,
    Tooltip,
} from 'recharts';
import { ScreenerStock } from '../types';

interface Props {
    stock: ScreenerStock;
    onClose: () => void;
}

const TIER_STYLE: Record<string, string> = {
    A: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    B: 'text-sky-400 bg-sky-400/10 border-sky-400/30',
    C: 'text-slate-300 bg-slate-400/10 border-slate-400/30',
    D: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    F: 'text-rose-400 bg-rose-400/10 border-rose-400/30',
};

/** 각 지표를 0~100 "우수도" 점수로 정규화 (높을수록 좋음) */
function normalizeMetrics(f: ScreenerStock['fundamentals']) {
    return [
        {
            metric: 'PER',
            value: f.per > 0 ? Math.max(0, Math.min(100, 100 - f.per * 1.5)) : 0,
            raw: f.per > 0 ? `${f.per.toFixed(1)}배` : 'N/A',
        },
        {
            metric: 'ROE',
            value: Math.max(0, Math.min(100, f.roe)),
            raw: `${f.roe.toFixed(1)}%`,
        },
        {
            metric: '배당',
            value: Math.min(100, f.dividendYield * 20),
            raw: `${f.dividendYield.toFixed(2)}%`,
        },
        {
            metric: '안전성',
            value: Math.max(0, 100 - f.debtRatio * 0.25),
            raw: `부채 ${f.debtRatio.toFixed(0)}%`,
        },
        {
            metric: 'PBR',
            value: f.pbr > 0 ? Math.max(0, Math.min(100, 100 - f.pbr * 15)) : 0,
            raw: f.pbr > 0 ? `${f.pbr.toFixed(2)}배` : 'N/A',
        },
        {
            metric: 'EPS',
            value: f.eps > 0 ? Math.min(100, (f.eps / 5000) * 100) : 0,
            raw: f.eps > 0 ? `₩${f.eps.toLocaleString()}` : 'N/A',
        },
    ];
}

const FundRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div className="flex justify-between items-center py-1.5 border-b border-slate-700/50 last:border-0">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-slate-200">{value}</span>
    </div>
);

export const StockDetailCard: React.FC<Props> = ({ stock, onClose }) => {
    const metrics = normalizeMetrics(stock.fundamentals);

    return (
        /* Backdrop */
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={onClose}
        >
            <div
                className="w-full max-w-lg bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-start justify-between px-5 py-4 border-b border-slate-700 bg-slate-800">
                    <div className="flex items-center gap-3">
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="text-lg font-bold text-white font-mono">{stock.ticker}</span>
                                <span className={`px-2 py-0.5 text-xs font-bold rounded border ${TIER_STYLE[stock.tier] ?? TIER_STYLE['C']}`}>
                                    {stock.tier}
                                </span>
                                {stock.isAnomaly && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded bg-amber-400/10 border border-amber-400/30 text-amber-400">
                                        <AlertTriangle className="w-3 h-3" /> 이상
                                    </span>
                                )}
                            </div>
                            <p className="text-sm text-slate-400 mt-0.5">{stock.name || '—'}</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-200 transition-colors mt-0.5">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-5 space-y-5">
                    {/* 점수 요약 */}
                    <div className="grid grid-cols-4 gap-3">
                        {[
                            { label: '군집', value: `C${stock.clusterId}` },
                            { label: '기술점수', value: stock.techScore.toFixed(1) },
                            { label: '재무점수', value: stock.fundScore.toFixed(1) },
                            { label: '종합점수', value: stock.totalScore.toFixed(1) },
                        ].map((item) => (
                            <div key={item.label} className="bg-slate-800 rounded-lg p-3 text-center border border-slate-700">
                                <p className="text-xs text-slate-500 mb-1">{item.label}</p>
                                <p className="text-base font-bold text-white font-mono">{item.value}</p>
                            </div>
                        ))}
                    </div>

                    {/* 레이더 차트 */}
                    <div>
                        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">재무 지표 (0–100 우수도)</h4>
                        <div className="h-52">
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart data={metrics} outerRadius="70%">
                                    <PolarGrid stroke="#334155" />
                                    <PolarAngleAxis
                                        dataKey="metric"
                                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                                    />
                                    <Radar
                                        dataKey="value"
                                        stroke="#6366f1"
                                        fill="#6366f1"
                                        fillOpacity={0.25}
                                        strokeWidth={2}
                                    />
                                    <Tooltip
                                        formatter={(value: number, _: string, entry: any) => [
                                            `${value.toFixed(1)} (${entry.payload.raw})`,
                                            entry.payload.metric,
                                        ]}
                                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: 12 }}
                                        labelStyle={{ display: 'none' }}
                                    />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* 재무 수치 */}
                    <div>
                        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">기본적분석 지표</h4>
                        <div className="bg-slate-800 rounded-lg px-4 py-1 border border-slate-700">
                            <FundRow label="PER (주가수익비율)" value={stock.fundamentals.per > 0 ? `${stock.fundamentals.per.toFixed(1)}배` : 'N/A'} />
                            <FundRow label="ROE (자기자본이익률)" value={`${stock.fundamentals.roe.toFixed(1)}%`} />
                            <FundRow label="배당수익률" value={`${stock.fundamentals.dividendYield.toFixed(2)}%`} />
                            <FundRow label="부채비율" value={`${stock.fundamentals.debtRatio.toFixed(0)}%`} />
                            <FundRow label="PBR (주가순자산비율)" value={stock.fundamentals.pbr > 0 ? `${stock.fundamentals.pbr.toFixed(2)}배` : 'N/A'} />
                            <FundRow label="EPS (주당순이익)" value={stock.fundamentals.eps > 0 ? `₩${stock.fundamentals.eps.toLocaleString()}` : 'N/A'} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
