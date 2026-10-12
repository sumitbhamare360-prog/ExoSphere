# L3 real-data validation: WASP-39 b NIRSpec PRISM

Generated: 2026-10-12 11:23:55
Analyses: EXO-000002, EXO-000003, EXO-000004

Model caveat (applies to every verdict below): V1 uses an isothermal,
constant-abundance, 5-gas mock-opacity model; published retrievals use
full radiative-convective-photochemical equilibrium grids with many
more species. Quantitative differences are expected and must be
explained, never hidden. No setting was tuned to force agreement.

## (a) Data round-trip (L1 recheck)
- PASS: 147 points round-trip exactly (max abs diff 0.0; archive sha 3359e371ec0364d3).

## (b) Molecules vs literature

| Molecule | Published | Ours | Grade |
|---|---|---|---|
| H2O | 33sigma PRISM (R+23); 21.5sigma G395H (A+23) | ln B = 196.2 | PASS: supported (ln B = 196.2) |
| CO2 | 28sigma PRISM (R+23); 28.5sigma G395H (A+23) | ln B = 191.0 | PASS: supported (ln B = 191.0) |
| CO | 7sigma PRISM (R+23) | ln B = 9.3 | SKIP: POOR data quality: no claim |
| CH4 | absent; 3sigma UL 5e-6 (R+23) | ln B = 0.6 | PASS: not constrained, as published |
| SO2 | 2.7sigma PRISM (R+23); 4.8sigma G395H (A+23); MIRI 0.5-25 ppm (P+24) | ln B = -3.5 | PASS: not constrained (ln B = -3.5) |

Section verdict: PARTIAL: assessable molecules agree with literature; CO is gated by its POOR band S/N (1.63, flat windows) per AGENTS.md rule 3, while playing only a marginal joint-fit role (best-fit removal 3.4 nats, nested ln B ~8-9 within systematics). No CO claim made.


## (c) Parameters vs published

- T: FAIL: 95% CI [538, 778] K outside 800-1500 K
- CH4_UL: PASS: 95% UL -5.00 dex (published 3sigma UL -5.3 dex)
- metallicity: PASS: proxy 2.2x solar within 1-300x published span
- C_O: PASS: proxy 0.39 within published 0.1-2.0 span
- derived_note: metallicity proxy 2.2x solar, C/O proxy 0.39 (order-of-magnitude only; 5-gas constant profiles vs RCPE grids)
- modes_note: Posterior modes by seed (documented T/cloud/abundance degeneracy): 42: T median 669 K, 95% CI [538, 778]; 123: T median 661 K, 95% CI [535, 776]; 7: T median 1872 K, 95% CI [1282, 2222]

## (d) Fit quality
- reduced chi2 = 41.05 (chi2 = 5706, dof = 139); worst bins (sigma, um): [(18.3, 2.694), (17.9, 2.823), (15.2, 2.65)]

## (e) ML vs retrieval
- ML candidate scores top: [('H2O', 0.0), ('CO2', 0.0)] (ML candidate score; not abundances or detections). No undiscussed contradiction enforced by report wording tests.

## Seed stability
- FAIL: 9/24 within even 95% CIs

## References

- Rustamkulov et al. 2023, Nature 614, 659 (JWST NIRSpec PRISM)
- Carter et al. 2024, Nature Astronomy 8, 1008 (benchmark)
- Alderson et al. 2023 (JWST NIRSpec G395H)
- Feinstein et al. 2023 (JWST NIRISS/SOSS)
- Powell et al. 2024 (JWST MIRI LRS)
- Moran et al. 2024, arXiv:2405.02656 (information content)

## Overall

- FAIL (see failing criteria above)
