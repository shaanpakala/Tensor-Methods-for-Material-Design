# Context for a new agent

This repo is the code for the published KDD 2026 paper **Tensor Methods: A Unified and Interpretable Approach for Material Design**. The GitHub link in the paper is https://github.com/shaanpakala/Tensor-Methods-for-Material-Design. Anyone who follows it should still find the original experiment notebooks.

The paper is the source for the science. Read `.gitignore/paper.pdf` before stating a result. DOI: https://doi.org/10.1145/3770855.3819008. Contact: `spaka002@ucr.edu`.

The idea, in one line: each design variable is a mode of a tensor, each tested design is a filled entry, and completing the tensor predicts the rest. The factor matrices are the interpretation. `README.md` maps the experiments to notebooks.

## What you may edit

The published experiment files are commit `0d521979b80ec2e6a4bd3e9d19f76a9626a5cd20` (24 July 2026):

https://github.com/shaanpakala/Tensor-Methods-for-Material-Design/tree/0d521979b80ec2e6a4bd3e9d19f76a9626a5cd20

Do not edit, delete, or add files inside these paths. Read them, import them, and run them. Put this rule in any sub-agent prompt.

- `interpret.ipynb`, `sl_table.ipynb`, `biased_table.ipynb`, `biased.ipynb`, `biased_levels.ipynb`
- `data/`
- `tensor_completion_models/`

`main` has moved since then. `demo/` and the root `README.md` were added or replaced afterward, and both may be edited. Other new work is a new top-level file or directory, or anything under `.gitignore/`.

`.gitignore/` is local only. It holds this note and `paper.pdf`. There is no `plans/` directory and no `papers/` directory.

## Where things are

`README.md` is the public map from paper section 3 to the five notebooks. `demo/` is a short public notebook with its own `requirements.txt` and `demo/data/`. The saved demo run is the crossed-barrel table. `tensor_completion_models/` is the library the experiment notebooks import. Run those notebooks from the repo root.

`codes_costco_tucker/` is an older trainer, not the entry point for the paper figures.

## Data

Lattice data is not in the repo. The notebooks expect `data/database_latticedesign.csv` and open on that dataset. Do not invent a stand-in.

Crossed barrel is `data/crossed_barrel_cleaned.csv`. The experiment notebooks still ask for `data/crossed_barrel_dataset_v2.csv` and a column named `toughness`. Leave those notebooks alone. `demo/` reads the cleaned file.

Cogni-e-Spin is `data/Cogni-e-SpinDB 1.0.csv`, and that path matches the notebooks. It is the dataset that loads as committed.
