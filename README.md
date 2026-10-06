# Tensor Methods: A Unified and Interpretable Approach for Material Design

Code for the KDD'26 AI for Sciences track paper

**Paper:** [https://doi.org/10.1145/3770855.3819008](https://doi.org/10.1145/3770855.3819008)

**Contact:** `spaka002@ucr.edu`

## Demo: Try a decomposition

`[demo/](demo/)` is a short notebook you can run without the rest of the experiment code. Put a CSV in `demo/data/`. Each row is one tested design: the design variables and one property. The notebook lists the CSVs, fits a CP decomposition on the observed rows, draws a parity plot on a 20% holdout, and plots two modes you choose.

```bash
cd demo
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

The saved run is the crossed-barrel table at rank 4, the setting used for the factor plots in Section 3.1. The plotted modes are twist (`theta`) and radius (`r`).

## Reproduce Section 3

Run the notebooks from the repository root. They import `tensor_completion_models` as a top-level package.


| Notebook              | Paper  | What it produces                                                                                    |
| --------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| `interpret.ipynb`     | §3.1   | CPD factor plots. Lattice rank 3, crossed barrel rank 4, Cogni-e-Spin rank 3.                       |
| `sl_table.ipynb`      | §3.2   | Uniform 80/20 table (Figure 6). Mean and standard deviation over 10 runs.                           |
| `biased_table.ipynb`  | §3.3.2 | Aggregate metrics under biased sampling (Figure 8) and the t-tests (Figure 9).                      |
| `biased.ipynb`        | §3.3.2 | MAE by region of the two biased design parameters (Figure 10).                                      |
| `biased_levels.ipynb` | §3.3.3 | R² and MAE on out-of-distribution points as the number of those training samples grows (Figure 11). |


Each notebook has a `dataset` switch: `'lattice'`, `'crossed_barrel'`, or `'cogni_spin'`. The lattice dataset is not available in this repo. Ranks and learning rates for the paper are in the training cells. Biased sampling is built inside the notebooks. It is not a column in the CSVs. Targets are scaled to [0, 1] before the tables, so MAE and RMSE are unitless.

*Please note this* `README.md` *file was created with heavy usage of cursor.ai, apologies for any mistakes.*

### Citation:

```
@inproceedings{pakala2026tensor,
  title={Tensor Methods: A Unified and Interpretable Approach for Material Design},
  author={Pakala, Shaan and Gongora, Aldair E and Giera, Brian and Papalexakis, Evangelos E},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={11740--11749},
  year={2026}
}
```

