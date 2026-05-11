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

To run a toy config with a linear annealing schedule for `T`:

```bash
python -m nets.experiments.train --config-name linear_scheduler_toy_cpu
```

To run asymmetric two-mode examples with mixture weights `1/3` and `2/3`:

```bash
python -m nets.experiments.train --config-name asymmetric_two_mode_1d
python -m nets.experiments.train --config-name asymmetric_two_mode_2d
```

To run a configurable multi-mode 2D example with explicit mode weights, mode
covariances, and a minimum distance between any two modes:

```bash
python -m nets.experiments.train --config-name constrained_random_modes_2d
```

To run multiple seeds, evaluate each best checkpoint, log aggregate metrics, and
save high-dimensional sample cross-section plots:

```bash
python -m nets.experiments.train_multiseed --config-name constrained_random_modes_2d \
  x_dim=16 \
  'target_mode_weights=[0.5,0.5]' \
  'target_mode_covariances=[1.0,6.0]' \
  'multiseed_seeds=[0,1,2,3,4]' \
  plot_every_n_epochs=0 \
  eval_plot_num_trajectories=128 \
  eval_plot_min_points=4000 \
  cross_section_max_pairs=6 \
  marginal_max_dims=8
```

The multiseed runner writes per-seed and aggregate files to:

```bash
checkpoints/<run_group>/<run_name>_multiseed/multiseed_metrics.jsonl
checkpoints/<run_group>/<run_name>_multiseed/multiseed_metrics.csv
checkpoints/<run_group>/<run_name>_multiseed/multiseed_summary.csv
checkpoints/<run_group>/<run_name>_multiseed/seed_<seed>/eval_plots/
```

To run a warm-start Gaussian-mixture prior with exponentially decaying shared
variance and a density path based on `prior_potential + t * target_potential`:

```bash
python -m nets.experiments.train --config-name warmstart_prior_constrained_modes_2d
```

To add a learned `t(1-t)` potential on top of that warm-start prior path:

```bash
python -m nets.experiments.train --config-name learnable_warmstart_prior_constrained_modes_2d
python -m nets.experiments.train --config-name learnable_warmstart_prior_constrained_modes_2d_fixed_beta
```

You can also use Hydra overrides from the CLI:

```bash
python -m nets.experiments.train max_epochs=5 train_steps_per_epoch=10 run_name=debug_run
```

For the standalone linear scheduler, use:

```bash
python -m nets.experiments.train annealing_scheduler=linear start_T=0.1 dT=0.1 epochs_per_T=10
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
