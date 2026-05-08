# Cross-config paired tests

Per-config paired t-test (seed-paired) between residual-aware and baseline methods, then summarized across configs.

## Duffing

| comparison | metric | configs | wins (p<.05) | median p | mean diff |
|---|---|---|---|---|---|
| Taylor-analytic N=2 vs GMM-baseline N=2 (residual-aware Taylor vs GMM-baseline) | `one_step` | 1 | 0 / 1 | 0.4139 | +0.3457 |
| Taylor-analytic N=2 vs GMM-baseline N=2 (residual-aware Taylor vs GMM-baseline) | `r5s` | 1 | 0 / 1 | 0.3990 | +20.1529 |
| Taylor-analytic N=4 vs GMM-baseline N=4 (residual-aware Taylor vs GMM-baseline) | `one_step` | 1 | 0 / 1 | 0.9809 | -0.0152 |
| Taylor-analytic N=4 vs GMM-baseline N=4 (residual-aware Taylor vs GMM-baseline) | `r5s` | 1 | 0 / 1 | 0.4421 | +18.4897 |
| Taylor-analytic N=4 vs EDMD-disc deg=2 (Taylor-analytic (largest N) vs Global EDMD (lowest deg)) | `one_step` | 1 | 0 / 1 | 0.2889 | +0.9054 |
| Taylor-analytic N=4 vs EDMD-disc deg=2 (Taylor-analytic (largest N) vs Global EDMD (lowest deg)) | `r5s` | 1 | 0 / 1 | 0.4252 | +19.0761 |
| Local-EDMD-disc d2 N=4 vs EDMD-disc deg=2 (local-EDMD (largest N) vs Global EDMD (lowest deg)) | `one_step` | 1 | 1 / 1 | 0.0004 | -0.0492 |
| Local-EDMD-disc d2 N=4 vs EDMD-disc deg=2 (local-EDMD (largest N) vs Global EDMD (lowest deg)) | `r5s` | 1 | 1 / 1 | 0.0167 | -0.6034 |

## Lorenz

| comparison | metric | configs | wins (p<.05) | median p | mean diff |
|---|---|---|---|---|---|
| Taylor N=5 vs GMM N=5 (residual-aware Taylor vs GMM-baseline) | `one_step` | 1 | 1 / 1 | 0.0370 | -0.0429 |
| Taylor N=12 vs GMM N=12 (residual-aware Taylor vs GMM-baseline) | `one_step` | 1 | 0 / 1 | 0.0995 | -0.1599 |
| Taylor N=5 vs EDMD-disc deg-2 (Taylor-analytic (largest N) vs Global EDMD (lowest deg)) | `one_step` | 1 | 0 / 1 | 0.0411 | +0.7955 |
| Local-EDMD-disc N=5 vs EDMD-disc deg-2 (local-EDMD (largest N) vs Global EDMD (lowest deg)) | `one_step` | 1 | 0 / 1 | 0.1610 | +0.2985 |

## Pendulum

| comparison | metric | configs | wins (p<.05) | median p | mean diff |
|---|---|---|---|---|---|
| Taylor-analytic N=4 vs EDMD-disc deg=2 (Taylor-analytic (largest N) vs Global EDMD (lowest deg)) | `one_step` | 1 | 1 / 1 | 0.0223 | -0.0433 |
| Taylor-analytic N=4 vs EDMD-disc deg=2 (Taylor-analytic (largest N) vs Global EDMD (lowest deg)) | `r5s` | 1 | 0 / 1 | 0.0900 | -1.3760 |
| Local-EDMD-disc N=4 vs EDMD-disc deg=2 (local-EDMD (largest N) vs Global EDMD (lowest deg)) | `one_step` | 1 | 0 / 1 | 0.1662 | -0.0845 |
| Local-EDMD-disc N=4 vs EDMD-disc deg=2 (local-EDMD (largest N) vs Global EDMD (lowest deg)) | `r5s` | 1 | 1 / 1 | 0.0429 | -1.5792 |
