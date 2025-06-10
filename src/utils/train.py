import os
import random
import tempfile
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

class PostBurninCheckpoint(ModelCheckpoint):
    def __init__(self, burn_in_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.burn_in_epochs = burn_in_epochs

    def on_validation_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch >= self.burn_in_epochs:
            super().on_validation_end(trainer, pl_module)

def train_module(
    module: pl.LightningModule,
    run_name: str,
    run_group: str = "test",
    max_epochs: int = 500,
    val_freq: int = 1,
    wandb: bool = False,
    wandb_project: Optional[str] = None,
    num_devices: int = 1,
    checkpoint: Optional[str] = None,
    mute_output: Optional[bool] = False,
    checkpoint_burn_in_epochs: int = 0,
):
    id = random.randint(0, 10000000)
    run_name = f"{run_name}_{id}"

    if mute_output:
        temp_dir = tempfile.TemporaryDirectory()
        checkpoints_dir = temp_dir.name
    else:
        curr_dir = os.getcwd()
        checkpoints_dir = os.path.join(curr_dir, "checkpoints", run_group)
        
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    print(f"Checkpoint directory: {checkpoints_dir}")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")


    checkpoint_callback = PostBurninCheckpoint(
        burn_in_epochs=checkpoint_burn_in_epochs,
        dirpath=os.path.join(checkpoints_dir, run_name),
        filename="{epoch:02d}-{val_w2:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="val_w2",
        every_n_epochs=1,
        mode="min",
    )

    if num_devices > 1:
        raise NotImplementedError(
            "Multi-GPU training is not implemented yet. Please use a single GPU for now."
        )

    trainer_kwargs = {
        "accelerator": "gpu",
        "callbacks": [checkpoint_callback, lr_monitor],
        "max_epochs": max_epochs,
        "devices": num_devices,
        "default_root_dir": checkpoints_dir,
        "check_val_every_n_epoch": val_freq,
    }

    if wandb:
        assert wandb_project is not None, "Wandb project must be specified if wandb is enabled."
        wandb_logger = WandbLogger(
            project=wandb_project,
            group=run_group,
            name=run_name,
            save_dir=checkpoints_dir,
        )
        trainer_kwargs["logger"] = wandb_logger
        
    trainer = pl.Trainer(**trainer_kwargs)
    if checkpoint is not None:
        trainer.fit(module, ckpt_path=checkpoint)
    else:
        trainer.fit(module)

    if mute_output:
        temp_dir.cleanup()
