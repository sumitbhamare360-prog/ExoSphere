import { createContext, useContext, useCallback } from 'react';
import type { ReactNode } from 'react';
import { useExoStore } from '../store/useExoStore';

interface ExoContextValue {
  setSelectedMolecule: (molecule: string | null) => void;
  setSelectedWavelength: (wavelength: number | null) => void;
  setSelectedRegion: (region: { start: number; end: number } | null) => void;
  setSelectedWavelengthRegion: (region: { start: number; end: number; molecule?: string } | null) => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setActiveResultTab: (tab: string) => void;
  setMoleculeBands: (bands: Record<string, Array<{ start: number; end: number }>>) => void;
  setWavelengthRange: (range: { min: number; max: number } | null) => void;
  resetSelection: () => void;
  resetAll: () => void;
}

const ExoContext = createContext<ExoContextValue | null>(null);

interface ExoProviderProps {
  children: ReactNode;
}

export function ExoProvider({ children }: ExoProviderProps) {
  const setSelectedMolecule = useExoStore((s) => s.setSelectedMolecule);
  const setSelectedWavelength = useExoStore((s) => s.setSelectedWavelength);
  const setSelectedRegion = useExoStore((s) => s.setSelectedRegion);
  const setSelectedWavelengthRegion = useExoStore((s) => s.setSelectedWavelengthRegion);
  const setSidebarOpen = useExoStore((s) => s.setSidebarOpen);
  const toggleSidebar = useExoStore((s) => s.toggleSidebar);
  const setActiveResultTab = useExoStore((s) => s.setActiveResultTab);
  const setMoleculeBands = useExoStore((s) => s.setMoleculeBands);
  const setWavelengthRange = useExoStore((s) => s.setWavelengthRange);
  const resetSelection = useExoStore((s) => s.resetSelection);
  const resetAll = useExoStore((s) => s.resetAll);

  const value: ExoContextValue = {
    setSelectedMolecule: useCallback((molecule: string | null) => setSelectedMolecule(molecule), [setSelectedMolecule]),
    setSelectedWavelength: useCallback(
      (wavelength: number | null) => setSelectedWavelength(wavelength),
      [setSelectedWavelength]
    ),
    setSelectedRegion: useCallback(
      (region: { start: number; end: number } | null) => setSelectedRegion(region),
      [setSelectedRegion]
    ),
    setSelectedWavelengthRegion: useCallback(
      (region: { start: number; end: number; molecule?: string } | null) =>
        setSelectedWavelengthRegion(region),
      [setSelectedWavelengthRegion]
    ),
    setSidebarOpen: useCallback((open: boolean) => setSidebarOpen(open), [setSidebarOpen]),
    toggleSidebar: useCallback(() => toggleSidebar(), [toggleSidebar]),
    setActiveResultTab: useCallback((tab: string) => setActiveResultTab(tab), [setActiveResultTab]),
    setMoleculeBands: useCallback(
      (bands: Record<string, Array<{ start: number; end: number }>>) => setMoleculeBands(bands),
      [setMoleculeBands]
    ),
    setWavelengthRange: useCallback(
      (range: { min: number; max: number } | null) => setWavelengthRange(range),
      [setWavelengthRange]
    ),
    resetSelection: useCallback(() => resetSelection(), [resetSelection]),
    resetAll: useCallback(() => resetAll(), [resetAll]),
  };

  return <ExoContext.Provider value={value}>{children}</ExoContext.Provider>;
}

export function useExoContext(): ExoContextValue {
  const context = useContext(ExoContext);
  if (!context) {
    throw new Error('useExoContext must be used within an ExoProvider');
  }
  return context;
}
