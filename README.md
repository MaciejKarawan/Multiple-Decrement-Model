# Multiple-Decrement Life Insurance Model

**Actuarial pricing of a 30-year term life policy with cause-dependent benefits. Mortality is calibrated to the Polish 2023 life tables (TTŻ 2023), and premiums and reserves are compared under three mortality models.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-optimize-8CAAE6?logo=scipy)
![Topic](https://img.shields.io/badge/topic-actuarial%20science-blueviolet)

## The product

A policy for a 25-year-old over n = 30 years at an effective rate i = 5%. The death benefit depends on the cause of death (a **multiple-decrement** setting):

| Cause | Share of mortality intensity | Benefit |
|---|---|---|
| Natural death | 85% | 100,000 PLN |
| Accident | 10% | 200,000 PLN |
| Disease | 5% | 500,000 PLN |

## Mortality models

| Model | Survival function | Calibration |
|---|---|---|
| **A: constant force** | `S(t) = exp(−μt)` | least squares on an age window (25–55, plus 20–60 and 15–65 for comparison) |
| **B: Makeham** | `S(t) = exp(−At − B/ln c · (c^{x+t} − c^x))` | least squares (L-BFGS-B) on the whole table |
| **C: empirical** | directly from TTŻ 2023 | none needed |

Each model is fitted separately for men and women using the official Polish 2023 life tables (`data/raw/ttz2023.xlsx`).

## What is computed

- Yearly death probabilities per cause, `q_j` and cumulative `Q_j`
- Expected present value of benefits **A** and of the premium annuity **ä**, giving the **level annual net premium**
- **Prospective reserves** over the policy term
- Split of each premium into a **risk premium** and a **savings premium**
- Side-by-side comparison of all quantities for models A, B and C and for both sexes

## Repository structure

```
src/decrement_model/   model.py: survival models, calibration, EPVs, premiums,
                       reserves, risk/savings split, plots
scripts/run_analysis   part 1: calibration and fit plots; part 2: premiums, reserves, figures
data/raw/              TTŻ 2023 life tables
```

## How to run

```bash
python -m pip install -e .
python -m scripts.run_analysis
```

Figures and the premium summary are written to `results/generated/`. Smoke tests: `python -m unittest discover -s tests`.

> **Note:** in the current script the policy parameters for part 2 (rate, benefits, cause shares) are inside a commented-out block, and the female constant-force fit reuses the male table. Part 1 runs as is. Part 2 needs those two lines restored.
