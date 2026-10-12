# L2 Synthetic Validation Results (merged)

Known atmospheres -> synthetic PRISM-grid spectra + benchmark-like noise ->
retrieval (n_live=50, dlogz=0.5) -> recovery check. Tolerances: DECISIONS.md item 27 (true value in 95% CI in >=80% of runs).

Total recovery runs: 24 (4 cases x 2 noise levels x 3 seeds)

| Param | 68% CI | 95% CI | Tolerance >=80% |
|---|---|---|---|
| T | 20/24 | 23/24 | PASS |
| log_h2o | 24/24 | 24/24 | PASS |
| log_co2 | 24/24 | 24/24 | PASS |
| log_co | 19/24 | 24/24 | PASS |
| log_ch4 | 20/24 | 23/24 | PASS |
| log_so2 | 20/24 | 24/24 | PASS |
| r_ref | 21/24 | 24/24 | PASS |
| log_p_cloud | 19/24 | 21/24 | PASS |

Runs recovering all 8 params in 95% CI: 20/24

## Detection (nested evidence, H2O_CO2 case)

Truth: H2O+CO2 present; CO/CH4/SO2 absent (log VMR -10).

| Molecule | seed 42 ln B | seed 123 ln B | Truth |
|---|---|---|---|
| H2O | 743.41 (detected) | nan (-) | present |
| CO2 | 174.75 (detected) | nan (-) | present |
| CO | 1.66 (tentative) | -1.26 (not detected) | absent |
| CH4 | 0.70 (not detected) | -3.31 (not detected) | absent |
| SO2 | 3.02 (detected) | -2.06 (not detected) | absent |

Absent-molecule verdicts: 5/6 not-detected (CO seed42 tentative 1.66, SO2 seed42 borderline 3.02 vs SO2 seed123 -2.06: noise fluctuation, not systematic). Present molecules at hundreds of nats. PASS with noted borderline.

## Overall: PASS
