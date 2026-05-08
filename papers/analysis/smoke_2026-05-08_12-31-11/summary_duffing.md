# Duffing -- multi-config summary

Mean +/- 95% CI across seeds for each (method, config, metric).
Metrics included: `one_step`, `r5s`, `r1s`.

## metric: `one_step`

| method | smoke_duffing |
|---|---|
| EDMD-disc deg=2 | 0.0516 +/- 0.0057 |
| GMM-baseline N=2 | 1.6678 +/- 1.5875 |
| GMM-baseline N=4 | 0.9722 +/- 0.3311 |
| Local-EDMD-disc d2 N=2 | 0.0119 +/- 0.0200 |
| Local-EDMD-disc d2 N=4 | 0.0025 +/- 0.0033 |
| Taylor-analytic N=2 | 2.0134 +/- 0.5692 |
| Taylor-analytic N=4 | 0.9571 +/- 2.7294 |

## metric: `r5s`

| method | smoke_duffing |
|---|---|
| EDMD-disc deg=2 | 0.7470 +/- 0.0633 |
| GMM-baseline N=2 | 39.9678 +/- 82.5767 |
| GMM-baseline N=4 | 1.3334 +/- 1.3389 |
| Local-EDMD-disc d2 N=2 | 0.5618 +/- 0.5365 |
| Local-EDMD-disc d2 N=4 | 0.1436 +/- 0.2768 |
| Taylor-analytic N=2 | 60.1207 +/- 1.3726 |
| Taylor-analytic N=4 | 19.8231 +/- 82.6104 |

## metric: `r1s`

| method | smoke_duffing |
|---|---|
| EDMD-disc deg=2 | 0.6451 +/- 0.1225 |
| GMM-baseline N=2 | 1.2107 +/- 1.1854 |
| GMM-baseline N=4 | 0.5459 +/- 0.1946 |
| Local-EDMD-disc d2 N=2 | 0.1567 +/- 0.1672 |
| Local-EDMD-disc d2 N=4 | 0.0341 +/- 0.0680 |
| Taylor-analytic N=2 | 1.4935 +/- 0.0093 |
| Taylor-analytic N=4 | 0.5665 +/- 1.9275 |
