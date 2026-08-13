# Reproduce Real Data Results

This folder contains the scripts and processed data needed to reproduce the real-data results for Table 2 and Figures 7-8.

## Environment

Use Python 3.9 or newer. Install the required packages:

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

The scripts expect the processed real data file to be in the current folder:

```text
processed_data.pkl
```

All result files are saved under:

```text
outputs/
```

## Table 2

Run:

```bash
python Table2.py
```

This performs leave-one-out evaluation over composite perturbations and saves:

```text
outputs/loocv_test_summary_with_time.xlsx
```

The Excel file contains:

- `summary_test`: Table 2 summary statistics and total runtime.
- `fold_metrics`: per-fold held-out test metrics.

## Figure 7

Run:

```bash
python Figure7.py
```

This saves:

```text
outputs/Figure7.pdf
```

## Figure 8

Run:

```bash
python Figure8.py
```

This saves:

```text
outputs/Figure8.pdf
```
