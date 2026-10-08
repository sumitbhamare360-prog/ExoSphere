import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface WavelengthRegion {
  start: number;
  end: number;
  molecule?: string;
}

export interface SelectedRegion {
  wavelengthStart: number;
  wavelengthEnd: number;
  molecules: string[];
}

export interface ExoStoreState {
  // Molecule selection state
  selectedMolecule: string | null;
  setSelectedMolecule: (molecule: string | null) => void;
  
  // Wavelength selection state
  selectedWavelength: number | null;
  setSelectedWavelength: (wavelength: number | null) => void;
  
  // Selected wavelength region (for highlighting)
  selectedRegion: { start: number; end: number } | null;
  setSelectedRegion: (region: { start: number; end: number } | null) => void;
  
  // Selected wavelength region from spectrum click
  selectedWavelengthRegion: { start: number; end: number; molecule?: string } | null;
  setSelectedWavelengthRegion: (region: { start: number; end: number; molecule?: string } | null) => void;
  
  // UI state
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  
  // Active tabs
  activeResultTab: string;
  setActiveResultTab: (tab: string) => void;
  
  // Molecule band windows (from API config)
  moleculeBands: Record<string, Array<{ start: number; end: number }>>;
  setMoleculeBands: (bands: Record<string, Array<{ start: number; end: number }>>) => void;
  
  // Wavelength range of current spectrum
  wavelengthRange: { min: number; max: number } | null;
  setWavelengthRange: (range: { min: number; max: number } | null) => void;
  
  // Reset functions
  resetSelection: () => void;
  resetAll: () => void;
}

const initialState = {
  selectedMolecule: null,
  selectedWavelength: null,
  selectedRegion: null,
  selectedWavelengthRegion: null,
  sidebarOpen: true,
  activeResultTab: 'spectrum',
  moleculeBands: {} as Record<string, Array<{ start: number; end: number }>>,
  wavelengthRange: null,
};

export const useExoStore = create<ExoStoreState>()(
  persist(
    (set) => ({
      ...initialState,
      
      setSelectedMolecule: (molecule) => set({ selectedMolecule: molecule }),
      
      setSelectedWavelength: (wavelength) => set({ selectedWavelength: wavelength }),
      
      setSelectedRegion: (region) => set({ selectedRegion: region }),
      
      setSelectedWavelengthRegion: (region) => set({ selectedWavelengthRegion: region }),
      
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      
      setActiveResultTab: (tab) => set({ activeResultTab: tab }),
      
      setMoleculeBands: (bands) => set({ moleculeBands: bands }),
      
      setWavelengthRange: (range) => set({ wavelengthRange: range }),
      
      resetSelection: () => set({
        selectedMolecule: null,
        selectedWavelength: null,
        selectedRegion: null,
        selectedWavelengthRegion: null,
      }),
      
      resetAll: () => set(initialState),
    }),
    {
      name: 'exosphere-store',
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        activeResultTab: state.activeResultTab,
        moleculeBands: state.moleculeBands,
      }),
    }
  )
);

// Helper to find molecules at a given wavelength
export function findMoleculesAtWavelength(
  wavelength: number,
  bands: Record<string, Array<{ start: number; end: number }>>
): string[] {
  return findAllMoleculesAtWavelength(wavelength, bands);
}

// Helper to find all molecules at a wavelength (including overlaps)
export function findAllMoleculesAtWavelength(
  wavelength: number,
  bands: Record<string, Array<{ start: number; end: number }>>
): string[] {
  const molecules: string[] = [];
  for (const [molecule, windows] of Object.entries(bands)) {
    for (const band of windows) {
      if (wavelength >= band.start && wavelength <= band.end) {
        molecules.push(molecule);
        break;
      }
    }
  }
  return molecules;
}

// Helper to check if wavelength falls in any molecule band
export function isWavelengthInAnyBand(
  wavelength: number,
  bands: Record<string, Array<{ start: number; end: number }>>
): boolean {
  for (const bands_ of Object.values(bands)) {
    for (const band of bands_) {
      if (wavelength >= band.start && wavelength <= band.end) {
        return true;
      }
    }
  }
  return false;
}