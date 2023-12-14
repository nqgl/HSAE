# based on and taken from https://github.com/ai-safety-foundation/sparse_autoencoder/blob/main/docs/content/demo.ipynb

from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    sweep,
    SweepConfig,
)
import wandb

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NOTEBOOK_NAME"] = "test.py"




sweep_config = SweepConfig(
    parameters=Hyperparameters(
        activation_resampler=ActivationResamplerHyperparameters(
            threshold_is_dead_portion_fires=Parameter(1e-6),
        ),
        loss=LossHyperparameters(
            l1_coefficient=Parameter(max=1e-2, min=4e-3),
        ),
        optimizer=OptimizerHyperparameters(
            lr=Parameter(max=1e-3, min=1e-5),
        ),
        source_model=SourceModelHyperparameters(
            name=Parameter("gelu-2l"),
            hook_site=Parameter("mlp_out"),
            hook_layer=Parameter(0),
            hook_dimension=Parameter(512),
        ),
        source_data=SourceDataHyperparameters(
            dataset_path=Parameter("NeelNanda/c4-code-tokenized-2b"),
        ),
    ),
    method=Method.RANDOM,
)


sweep(sweep_config=sweep_config)

wandb.finish()