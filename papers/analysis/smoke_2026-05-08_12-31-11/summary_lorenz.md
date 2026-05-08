# Lorenz -- multi-config summary

Mean +/- 95% CI across seeds for each (method, config, metric).
Metrics included: `one_step`, `r0_5s`, `r1s`.

## metric: `one_step`

| method | smoke_lorenz |
|---|---|
| EDMD-disc deg-2 | 0.0667 +/- 0.0373 |
| EDMD-pk deg-2 | 0.0667 +/- 0.0373 |
| GMM N=12 | 0.8510 +/- 0.7635 |
| GMM N=5 | 0.9052 +/- 0.7135 |
| Local-EDMD-disc N=12 | 0.1703 +/- 0.4435 |
| Local-EDMD-disc N=5 | 0.3652 +/- 0.6178 |
| Taylor N=12 | 0.6910 +/- 0.5542 |
| Taylor N=5 | 0.8622 +/- 0.7479 |

## metric: `r0_5s`

| method | smoke_lorenz |
|---|---|
| EDMD-disc deg-2 | 0.7976 +/- 1.2399 |
| GMM N=12 | 16.0064 +/- 5.2423 |
| GMM N=5 | 17.9428 +/- 4.4028 |
| Local-EDMD-disc N=12 | 0.5098 +/- 1.2848 |
| Local-EDMD-disc N=5 | 2.6023 +/- 3.6926 |
| Taylor N=12 | 15.4027 +/- 8.3400 |
| Taylor N=5 | 18.3666 +/- 9.8312 |

## metric: `r1s`

| method | smoke_lorenz |
|---|---|
| EDMD-disc deg-2 | 1.5837 +/- 0.7753 |
| GMM N=12 | 20.0035 +/- 15.2965 |
| GMM N=5 | 17.8219 +/- 13.9875 |
| Local-EDMD-disc N=12 | 2.0005 +/- 5.0915 |
| Local-EDMD-disc N=5 | 7.8261 +/- 16.9895 |
| Taylor N=12 | 17.3594 +/- 17.8314 |
| Taylor N=5 | 27.9991 +/- 17.0922 |
