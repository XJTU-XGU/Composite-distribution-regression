# Simulation Reproduction Guide

This folder contains the simulation scripts used to reproduce the figures and
table in the article. All commands below should be run from this directory.

```powershell
cd path\to\simulation
```

The scripts save figures to `outputs/`. Several long-running simulation results
have already been saved under `results/`, `results_null_scalar/`,
`results_nonlinear/`, and `results_got/`, so the figures can be regenerated
directly from the saved result files without rerunning all experiments.

## Environment

Use a Python environment with the required scientific-computing packages
installed, including:

- `numpy`
- `matplotlib`
- `torch`
- `scipy`

Our all results are generated using Python 3.9 on a CentOS 7 Linux server equipped with two Intel Xeon Gold 6246R processors
running at 3.40 GHz and 376 GB of system memory (RAM). Since the Linux system may not have "Times New Roman" font, one may
plot the figures on a Windows computer using the generated results. 

## Figure 1

Run:

```powershell
python Figure1.py
```

Output:

```text
outputs/Figure1.pdf
```

## Figure 2

Run:

```powershell
python Figure2.py
```

Output:

```text
outputs/Figure2.pdf
```

## Figure 3 and Table 1

Figure 3 and Table 1 use the grid experiments. To reproduce the data from
scratch, run the grid scripts:

```powershell
python Grid_d_N100.py
python Grid_N_d10.py
```

`Grid_d_N100.py` includes the GOT method and can take a long time. In particular, `d=50` takes an extremely
long time, one may run the code for it separately.  The
default saved results also include the high-dimensional setting `d=50`
(corresponding to `p=51` in the code parameter), which can be regenerated separately with:

```powershell
python Grid_d_N100.py --p-list 51 --output results/fig3_vector_decay_theta_no_intercept_results_50.mat
```

The corresponding results have already been saved in `results/`, so rerunning
these steps is optional if you only want to regenerate the figure and table.

Saved result files used by default:

```text
results/fig3_vector_decay_theta_no_intercept_results.mat
results/fig3_vector_decay_theta_no_intercept_results_50.mat
results/fig3_vector_decay_theta_no_intercept_N_grid_p11_results.mat
```

After the data are available, generate Figure 3 and print the computation-time
summary for Table 1:

```powershell
python Figure3_and_Table1.py
```

If you regenerated only a subset of the p-grid result files, pass the available
files explicitly, for example:

```powershell
python Figure3_and_Table1.py --p-grid-inputs results/fig3_vector_decay_theta_no_intercept_results.mat results/fig3_vector_decay_theta_no_intercept_results_50.mat
```

Output:

```text
outputs/Figure3.pdf
```

The Table 1 timing summary is printed in the terminal.

## Figures 4-6: Sensitivity Experiments

Figures 4-6 correspond to three sensitivity experiments. Each experiment has
one script for generating data and one script for plotting. The saved data are
already available in the corresponding result folders, so you can skip directly
to the plotting commands unless you need to rerun the simulations.

### Figure 4: Irrelevant Scalar Covariates

To regenerate the data:

```powershell
python Sensitivity_irrelevant_scalar.py
```

Saved result file:

```text
results_null_scalar/fig3_scalar_irrelevant_assume_d10_results.mat
```

To draw Figure 4:

```powershell
python Figure4.py
```

Output:

```text
outputs/Figure4.pdf
```

### Figure 5: Friedman Nonlinear Sensitivity

To regenerate the data:

```powershell
python Sensitivity_Friedman_phi.py
```

Saved result file:

```text
results_nonlinear/friedman10_structured_transport_pgfr_results.mat
```

To draw Figure 5:

```powershell
python Figure5.py
```

Output:

```text
outputs/Figure5.pdf
```

### Figure 6: GOT Sensitivity

To regenerate the data:

```powershell
python Sensitivity_GOT.py
```

This experiment is computationally expensive. The saved result file is:

```text
results_got/mixed_methods_p11_N100_beta_mu_fig3_noise_fixed_alpha.mat
```

To draw Figure 6:

```powershell
python Figure6.py
```

Output:

```text
outputs/Figure6.pdf
```

### Quick Reproduction from Saved Results

If you only want to regenerate the article figures and Table 1 from the saved
results, run:

```powershell
python Figure1.py
python Figure2.py
python Figure3_and_Table1.py
python Figure4.py
python Figure5.py
python Figure6.py
```

This will write the generated figures to `outputs/` and print the Table 1
timing summary when `Figure3_and_Table1.py` runs.
