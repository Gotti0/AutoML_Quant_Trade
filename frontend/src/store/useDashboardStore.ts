import { create } from 'zustand';
import { AlgorithmModel, RegimeProbability } from '../types';

interface DashboardState {
    algorithms: AlgorithmModel[];
    regimeHistory: RegimeProbability[];
    currentRegime: string | null;
    isLoading: boolean;
    error: string | null;

    setAlgorithms: (data: AlgorithmModel[]) => void;
    setRegimeHistory: (data: RegimeProbability[]) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
    algorithms: [],
    regimeHistory: [],
    currentRegime: null,
    isLoading: false,
    error: null,

    setAlgorithms: (data) => set({ algorithms: data }),
    setRegimeHistory: (data) => set({
        regimeHistory: data,
        currentRegime: data.length > 0 ? data[data.length - 1].dominantRegime : null
    }),
    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),
}));
