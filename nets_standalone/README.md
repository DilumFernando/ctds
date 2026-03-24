# NETS Standalone

Minimal standalone NETS package extracted from the parent repo.

This repo keeps only the essential learned-overdamped NETS path and replaces the
original PyTorch Lightning training stack with a pure PyTorch loop.

## Install

```bash
conda env create -f environment.yml
conda activate nets-standalone
pip install -e .
```

## Run

```bash
nets-train --config configs/learned_overdamped.toml --run-name learned_overdamped_test
```

You can also override config values from the CLI:

```bash
nets-train --config configs/learned_overdamped.toml --set max_epochs=5 --set train_steps_per_epoch=10
```
