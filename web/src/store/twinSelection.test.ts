import { beforeEach, describe, expect, it, vi } from 'vitest';

// zustand/persist touches localStorage, which does not exist in the node env.
const store: Record<string, string> = {};
vi.stubGlobal('localStorage', {
  getItem: (k: string) => store[k] ?? null,
  setItem: (k: string, v: string) => {
    store[k] = v;
  },
  removeItem: (k: string) => {
    delete store[k];
  },
});

import { useExoStore } from './useExoStore';

beforeEach(() => {
  useExoStore.getState().resetAll();
});

describe('molecule selection sync (dashboard <-> twin)', () => {
  it('starts with no molecule selected', () => {
    expect(useExoStore.getState().selectedMolecule).toBeNull();
  });

  it('selecting a molecule updates the shared store the twin subscribes to', () => {
    useExoStore.getState().setSelectedMolecule('CO2');
    expect(useExoStore.getState().selectedMolecule).toBe('CO2');
  });

  it('deselecting clears the highlight', () => {
    useExoStore.getState().setSelectedMolecule('H2O');
    useExoStore.getState().setSelectedMolecule(null);
    expect(useExoStore.getState().selectedMolecule).toBeNull();
  });

  it('stores molecule band windows shared with the spectrum viewer', () => {
    useExoStore.getState().setMoleculeBands({ CO2: [{ start: 4.2, end: 4.4 }] });
    expect(useExoStore.getState().moleculeBands.CO2).toEqual([{ start: 4.2, end: 4.4 }]);
  });
});
