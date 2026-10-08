import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const TWIN_DIR = join(process.cwd(), 'src', 'components', 'twin');

/** AGENTS.md rule 4: no surfaces/continents/oceans/photos, no molecule maps. */
const FORBIDDEN = [
  'TextureLoader',
  'useTexture',
  'useMatcapTexture',
  'useEnvironment',
  'continent',
  'ocean',
  'surface map',
  'surfacemap',
  // NOTE: bare 'photo' intentionally absent — the required disclaimer itself
  // says "Not a photograph", which would false-positive on a substring scan.
  'photograph of the planet',
  'molecule map',
];

/** Every twin view must carry the persistent measured/modelled disclaimer. */
const REQUIRED_DISCLAIMER_FRAGMENTS = ['Not a photograph'];

/** Negation words: a line that denies the concept is a disclaimer, not a claim. */
const NEGATION = /\b(no|not|never|without|avoid|against|isn't|don't|none)\b/i;

describe('twin visuals stay scientific (no fabricated geography)', () => {
  const files = readdirSync(TWIN_DIR).filter((f) => f.endsWith('.tsx') || f.endsWith('.ts'));

  it('twin components reference no textures or surface imagery', () => {
    const hits: string[] = [];
    for (const file of files) {
      if (file.endsWith('.test.ts')) continue;
      const text = readFileSync(join(TWIN_DIR, file), 'utf8');
      for (const [i, line] of text.split('\n').entries()) {
        if (NEGATION.test(line)) continue;
        const lower = line.toLowerCase();
        for (const token of FORBIDDEN) {
          if (lower.includes(token.toLowerCase())) hits.push(`${file}:${i + 1}: ${token}`);
        }
      }
    }
    expect(hits).toEqual([]);
  });

  it('keeps a persistent not-a-photograph disclaimer in the twin UI', () => {
    const joined = files
      .filter((f) => !f.endsWith('.test.ts'))
      .map((f) => readFileSync(join(TWIN_DIR, f), 'utf8'))
      .join('\n');
    for (const fragment of REQUIRED_DISCLAIMER_FRAGMENTS) {
      expect(joined).toContain(fragment);
    }
  });
});
