# How Copies Break Conformal Prediction

Reproduce the finite-sample copy-budget analysis and controlled regression experiments.

The study holds the predictor and test observations fixed while copying calibration records. The central result is an attained worst-case coverage frontier, together with a conservative correction when a valid upper bound on copied rows is known.

## Quick start

Use Python 3.12 in a fresh environment. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run_experiments.py --workers 10 --seeds 30
python src/analyze_results.py
python src/verify_results.py
```

On Windows, activate with `.venv\Scripts\activate`. Reduce `--workers` for fewer available CPU cores. The experiment uses one numerical thread per worker and CPU-based learners. Raw data snapshots are bundled; after dependency installation, the experiment runs offline. Use all 30 seeds to reproduce the reported summaries. Execution replaces numerical files in `results/`.

## Experimental design

Five tasks are created from UCI Airfoil Self-Noise, Concrete Compressive Strength, Energy Efficiency (heating load), and red- and white-Wine Quality. Exact full feature-target matches are collapsed before splitting to define finite empirical-support benchmarks. Original source-row indices are retained in the processed NPZ files. This target is a held-out support record, rather than a claim about physical-population sampling.

Each task uses 30 fixed seeds and a 60/20/20 train/calibration/test split (integer truncation for the first two sets). The three learners are standardized ridge regression, a 200-tree random forest, and 200-iteration histogram gradient boosting. Models are fitted once per split, with all copying interventions applied afterward. The 450 fitted models support 226,800 method-condition records; these records are correlated repeated measurements, not independent studies.

Added-copy ratios are 0, 0.25, 0.5, 1, 2, 4, and 9 per original calibration unit. Mechanisms are balanced, random, lowest-score fifth, highest-score fifth, lowest-first-feature fifth, and minimum-score allocation. Coverage targets are 80%, 90%, and 95%. The main copy-budget method receives the exact inserted copy count; a separate sensitivity analysis varies its assumed bound.

Summary standard errors use the 30 split-seed blocks after averaging datasets and models within each block. Synthetic experiments use 20,000 independent trials per distribution-size setting, with common draws across targets and copy budgets. Exhaustive checks cover 10,272 allocation cases. The appended-only copy model retains every original unit and requires exchangeability of original calibration and test scores conditional on training.

## Project contents

- `src/run_experiments.py`: checks raw archive hashes, prepares datasets, fits models, applies copying interventions, and runs score simulations.
- `src/analyze_results.py`: produces aggregate, dataset-level, paired-difference, and copy-bound sensitivity summaries.
- `src/verify_results.py`: validates experiment results against recorded numerical checksums and validates the saved design.
- `tests/test_calibration.py`: independent expanded-array and exhaustive allocation checks, copy-bound tests, ties, and infinity handling.
- `data/raw/`: four unchanged UCI source archives.
- `data/processed/`: five empirical-support datasets in CSV and compressed NPZ, including original source-row identifiers in NPZ.
- `data/download_manifest.json`: source URLs, file sizes, and SHA-256 hashes.
- `data/ATTRIBUTION.md`: creators, citation identifiers, source descriptions, and licenses.
- `results/metrics.csv`: coverage, width, relative width, finite-interval status, thresholds, and model RMSE for every condition.
- `results/predictions/`: train/calibration/test split identifiers, predictions, and residual arrays for all 450 fitted models.
- `results/synthetic.csv`: theoretical and empirical coverage, binomial Monte Carlo errors, and conditional-CDF checks.
- `results/summary.csv`, `dataset_summary.csv`, and `paired.csv`: numerical summaries.
- `results/budget_sensitivity.csv`: robustness to misspecified copy-count bounds.
- `results/checks.json`, `environment.json`, and `timings.csv`: invariant checks, dependency versions, and run timing records.
- `expected_checksums.json`: numerical result fingerprints from the verified reference execution.
- `CITATION.cff`: software citation information.

## Reference findings

At nominal 90% coverage and four added copies per original calibration record, lowest-error-fifth copying reduced average rowwise coverage to 50.40%, compared with 90.17% for source-ID calibration. The correctly specified budget method recovered the original intervals in this condition. Under random copying, rowwise coverage was 89.88%, while the budget correction increased average relative interval width to 1.6097. An effective-sample-size adjustment retained only 56.52% coverage under lowest-error copying. Concentrating copies at the minimum score caused that heuristic to return unbounded intervals.

Source-ID invariance and conservative budget protection hold under the stated exact-copy model. Identical measured values alone need not indicate copied observations. Near duplicates, deleted records, altered labels, and distribution shift require separate assumptions.

## Reproducibility and licenses

The reference environment is recorded in `results/environment.json`; direct dependencies are pinned in `requirements.txt`. Two complete executions produced identical main metrics. Exact floating-point fingerprints are environment-specific; `verify_results.py` checks the recorded numerical outputs, and the regression tests check mathematical behavior independently.

Original software and documentation are MIT licensed. Bundled UCI datasets retain their CC BY 4.0 license and attribution in `data/ATTRIBUTION.md`.
