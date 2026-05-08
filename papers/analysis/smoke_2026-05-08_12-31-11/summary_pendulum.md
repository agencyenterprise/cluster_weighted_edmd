# Pendulum -- multi-config summary

Mean +/- 95% CI across seeds for each (method, config, metric).
Metrics included: `one_step`, `r5s`, `r1s`.

## metric: `one_step`

| method | smoke_pendulum |
|---|---|
| EDMD-disc deg=2 | 0.1312 +/- 0.0349 |
| Local-EDMD-disc N=2 | 0.1272 +/- 0.0740 |
| Local-EDMD-disc N=4 | 0.0467 +/- 0.1967 |
| Taylor-analytic N=2 | 0.7619 +/- 0.8104 |
| Taylor-analytic N=4 | 0.0880 +/- 0.0574 |

## metric: `r5s`

| method | smoke_pendulum |
|---|---|
| EDMD-disc deg=2 | 2.1320 +/- 0.7893 |
| Local-EDMD-disc N=2 | 1.7680 +/- 1.6055 |
| Local-EDMD-disc N=4 | 0.5528 +/- 1.3841 |
| Taylor-analytic N=2 | 0.9846 +/- 0.3904 |
| Taylor-analytic N=4 | 0.7560 +/- 1.1772 |

## metric: `r1s`

| method | smoke_pendulum |
|---|---|
| EDMD-disc deg=2 | 1.1587 +/- 0.4283 |
| Local-EDMD-disc N=2 | 0.5011 +/- 0.4342 |
| Local-EDMD-disc N=4 | 0.0636 +/- 0.2180 |
| Taylor-analytic N=2 | 0.5729 +/- 0.2804 |
| Taylor-analytic N=4 | 0.1038 +/- 0.0958 |
