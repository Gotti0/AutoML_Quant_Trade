import React, { useState } from 'react';
import { ChevronUp, ChevronDown, AlertTriangle } from 'lucide-react';
import { ScreenerStock } from '../types';

type SortKey = keyof Pick<ScreenerStock, 'ticker' | 'name' | 'clusterId' | 'techScore' | 'fundScore' | 'totalScore' | 'tier'>;

interface Props {
    stocks: ScreenerStock[];
    onSelect: (stock: ScreenerStock) => void;
}

const TIER_STYLE: Record<string, string> = {
    A: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    B: 'text-sky-400 bg-sky-400/10 border-sky-400/30',
    C: 'text-slate-300 bg-slate-400/10 border-slate-400/30',
    D: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    F: 'text-rose-400 bg-rose-400/10 border-rose-400/30',
};

const PAGE_SIZE = 20;

export const ScreenerTable: React.FC<Props> = ({ stocks, onSelect }) => {
    const [sortKey, setSortKey] = useState<SortKey>('totalScore');
    const [sortAsc, setSortAsc] = useState(false);
    const [page, setPage] = useState(0);

    const handleSort = (key: SortKey) => {
        if (key === sortKey) {
            setSortAsc((p) => !p);
        } else {
            setSortKey(key);
            setSortAsc(false);
        }
        setPage(0);
    };

    const sorted = [...stocks].sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number);
        return sortAsc ? cmp : -cmp;
    });

    const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
    const pageSlice = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

    const SortIcon = ({ col }: { col: SortKey }) => {
        if (col !== sortKey) return <ChevronUp className="w-3 h-3 text-slate-600" />;
        return sortAsc
            ? <ChevronUp className="w-3 h-3 text-indigo-400" />
            : <ChevronDown className="w-3 h-3 text-indigo-400" />;
    };

    const Th = ({ children, col }: { children: React.ReactNode; col: SortKey }) => (
        <th
            className="px-3 py-2 text-left text-xs font-semibold text-slate-400 uppercase tracking-wide cursor-pointer hover:text-slate-200 select-none whitespace-nowrap"
            onClick={() => handleSort(col)}
        >
            <span className="inline-flex items-center gap-1">
                {children}
                <SortIcon col={col} />
            </span>
        </th>
    );

    return (
        <div className="flex flex-col gap-3">
            <div className="overflow-x-auto rounded-lg border border-slate-700">
                <table className="w-full text-sm">
                    <thead className="bg-slate-800/80">
                        <tr>
                            <th className="px-3 py-2 text-left text-xs font-semibold text-slate-400 uppercase tracking-wide w-6">#</th>
                            <Th col="ticker">코드</Th>
                            <Th col="name">종목명</Th>
                            <Th col="clusterId">군집</Th>
                            <Th col="techScore">기술점수</Th>
                            <Th col="fundScore">재무점수</Th>
                            <Th col="totalScore">종합점수</Th>
                            <Th col="tier">티어</Th>
                            <th className="px-3 py-2 text-left text-xs font-semibold text-slate-400 uppercase tracking-wide">이상</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {pageSlice.length === 0 && (
                            <tr>
                                <td colSpan={9} className="px-3 py-8 text-center text-slate-500 text-sm">
                                    표시할 종목이 없습니다.
                                </td>
                            </tr>
                        )}
                        {pageSlice.map((stock, idx) => (
                            <tr
                                key={stock.ticker}
                                onClick={() => onSelect(stock)}
                                className="hover:bg-slate-800/60 cursor-pointer transition-colors"
                            >
                                <td className="px-3 py-2 text-slate-500 text-xs">{page * PAGE_SIZE + idx + 1}</td>
                                <td className="px-3 py-2 font-mono text-slate-300 text-xs">{stock.ticker}</td>
                                <td className="px-3 py-2 text-slate-200 max-w-[140px] truncate">{stock.name || '—'}</td>
                                <td className="px-3 py-2 text-center">
                                    <span className="inline-block px-2 py-0.5 text-xs rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                                        C{stock.clusterId}
                                    </span>
                                </td>
                                <td className="px-3 py-2 text-slate-300 text-right font-mono">{stock.techScore.toFixed(1)}</td>
                                <td className="px-3 py-2 text-slate-300 text-right font-mono">{stock.fundScore.toFixed(1)}</td>
                                <td className="px-3 py-2 text-right font-mono font-semibold text-white">{stock.totalScore.toFixed(1)}</td>
                                <td className="px-3 py-2">
                                    <span className={`inline-block px-2 py-0.5 text-xs font-bold rounded border ${TIER_STYLE[stock.tier] ?? TIER_STYLE['C']}`}>
                                        {stock.tier}
                                    </span>
                                </td>
                                <td className="px-3 py-2 text-center">
                                    {stock.isAnomaly && (
                                        <AlertTriangle className="w-4 h-4 text-amber-400 inline-block" />
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between text-xs text-slate-500">
                <span>{stocks.length}개 종목</span>
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={page === 0}
                        className="px-2 py-1 rounded hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        이전
                    </button>
                    <span className="px-2">{page + 1} / {totalPages}</span>
                    <button
                        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                        disabled={page >= totalPages - 1}
                        className="px-2 py-1 rounded hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        다음
                    </button>
                </div>
            </div>
        </div>
    );
};
