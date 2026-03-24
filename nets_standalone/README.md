# NETS Standalone

Minimal standalone NETS package extracted from the parent repo.

This repo keeps only the essential learned-overdamped NETS path and replaces the
original PyTorch Lightning training stack with a pure PyTorch loop.

Configs are managed with Hydra and stored as YAML files in [nets/conf](/Users/dilumfernando/projects/ctds/nets_standalone/nets/conf).

## Install

```bash
conda env create -f environment.yml
conda activate nets-standalone
pip install -e .
```

## Run

```bash
python -m nets.experiments.train
```

That uses the default CPU toy config. To run the learned-overdamped config:

```bash
python -m nets.experiments.train --config-name learned_overdamped
```

To run a plain linear density path with a random GMM target:

```bash
python -m nets.experiments.train --config-name linear_random_gmm
```

To run asymmetric two-mode examples with mixture weights `1/3` and `2/3`:

```bash
python -m nets.experiments.train --config-name asymmetric_two_mode_1d
python -m nets.experiments.train --config-name asymmetric_two_mode_2d
```

You can also use Hydra overrides from the CLI:

```bash
python -m nets.experiments.train max_epochs=5 train_steps_per_epoch=10 run_name=debug_run
```

To log to WandB:

```bash
python -m nets.experiments.train wandb=true wandb_project=myproject
```

Each run writes epoch-by-epoch logs to:

```bash
checkpoints/<run_group>/<run_name_...>/metrics.jsonl
checkpoints/<run_group>/<run_name_...>/metrics.csv
checkpoints/<run_group>/<run_name_...>/train.log
checkpoints/<run_group>/<run_name_...>/plots/epoch_0010.png
```

These include `epoch`, `train_loss`, and, when validation runs, `val_loss` plus the
other validation metrics.

Trajectory plots are saved every `plot_every_n_epochs` epochs for 2D runs, with
particle positions colored by their weights. You can override this from Hydra:

```bash
python -m nets.experiments.train plot_every_n_epochs=5 plot_num_trajectories=48
```
