# Interpretability demo

Download this `demo/` folder by itself. Put a CSV of your own designs in `demo/data/`, install the packages below, and run `demo.ipynb`. Each row should be one tested design: the design variables and one measured property. The notebook fits a CP decomposition on those rows and returns a holdout parity plot plus factor plots for two variables you choose.

```bash
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

The notebook prints every CSV in `demo/data/` with an index. Set `data_file_i` to your file, then set `feature_columns`, `target_column`, and `modes_to_plot` to names in that file. A finished crossed-barrel run (toughness, twist, and radius) is already saved in the notebook. The charts follow Section 3.1 of [Tensor Methods: A Unified and Interpretable Approach for Material Design](https://doi.org/10.1145/3770855.3819008).

## What to edit


| Setting           | Role                                                                                           |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `data_file_i`     | Index of a CSV in `demo/data/`. The notebook prints that list first                            |
| `feature_columns` | Design variables. Each column becomes one tensor mode                                          |
| `target_column`   | Property stored in the observed entries                                                        |
| `modes_to_plot`   | Two names from `feature_columns`                                                               |
| `log_target`      | Log the property before scaling. Use this for a positive target that spans orders of magnitude |


The checked-in crossed-barrel file uses modes `n` (struts), `theta` (twist), `r` (radius), and `t` (thickness), with target `toughness_mean`. The saved factor plots are twist and radius. Any other file is used as provided, after dropping rows with missing values in the chosen columns. Choosing the Cogni-e-Spin CSV still applies that file's row cuts from the paper notebook.

Duplicate designs are averaged. Each distinct value in a column is mapped to an index, and those original values label the plot axes. The target is scaled to [0, 1]. Numeric modes are ordered from low to high. Other modes keep first-seen order.

Training uses rank 4 for the crossed-barrel example, Adam, mean absolute error, 3,000 epochs, and an 80/20 holdout. The parity plot is that 20% test set, with R² and MAE in the box. Factors are then divided by the L2 norm of each component and drawn as grouped bars.

*Please note this* `README.md` *file was created with heavy usage of cursor.ai, apologies for any mistakes.*