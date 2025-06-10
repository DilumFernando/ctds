# Continuously Tempered Diffusion Samplers
An implementation of [Continuously Tempered Diffusion Samplers](https://openreview.net/forum?id=060KGPDxbH) (CTDS), by Ezra Erives, Bowen Jing, Peter Holderrieth, and Tommi Jaakkola. This codebase also contains an implementation of [Non-Equilibrium Transport Sampler](https://arxiv.org/abs/2410.02711) (NETS), by Michael Albergo and Eric Vanden-Eijnden.

### CTDS
To train CTDS from scratch in a Jupyter notebook, see `notebooks/ctds.ipynb`. To play around with the pre-trained checkpoints, see `camera_ready.ipynb` and `extended_figures.ipynb`. To train from scratch on the command line, use e.g.,
```
CUDA_VISIBLE_DEVICES=0 RUN_CONFIG=learned_tempered python train_ctds.py 'ctds_config.run_name=learned_tempered_test'
```
This will train a CTDS model using the `.yaml` configuration file located at `/config/ctds_config/learned_tempered.yaml`. To train with Wandb, set `wandb=True` in the corresponding `.yaml` file and pass in `WANDB_PROJECT=...`, viz., 
```
CUDA_VISIBLE_DEVICES=0 RUN_CONFIG=learned_tempered WANDB_PROJECT=myproject python train_ctds.py 'ctds_config.run_name=learned_tempered_test'
```
An overview of available existing configuration files can be found at `/config/config.md`. 

### NETS
To train NETS from scratch in a Jupyter notebook, see `notebooks/nets.ipynb`. To play around with a pre-trained checkpoints, see `camera_ready.ipynb` and `extended_figures.ipynb`. To train from scratch on the command line, use e.g.,
```
CUDA_VISIBLE_DEVICES=0 RUN_CONFIG=learned_overdamped python train_nets.py 'nets_config.run_name=learned_overdamped_test'
```
This will train a NETS model using the `.yaml` configuration file located at `/config/nets_config/learned_overdamped.yaml`. You may similarly log to Wandb by following the directions given in the CTDS section. An overview of available existing configuration files can be found at `/config/config.md`. 
