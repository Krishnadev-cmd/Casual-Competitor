import base64
import io
import os
import warnings
from dataclasses import dataclass
import time

import fsspec
import json
import matplotlib.pyplot as plt
from narwhals import col
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from pprint import pprint
import streamlit as st

from pytorch_lightning.callbacks import TQDMProgressBar

from causica.datasets.causica_dataset_format import Variable, DataEnum, load_data
from causica.datasets.variable_types import VariableTypeEnum
from causica.datasets.tensordict_utils import tensordict_shapes
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule
from causica.sem.sem_distribution import SEMDistributionModule
from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig
from causica.distributions import (
    AdjacencyDistribution,
    ContinuousNoiseDist,
    DistributionModule,
    ENCOAdjacencyDistributionModule,
    GibbsDAGPrior,
    JointNoiseModule,
    create_noise_modules,
)
from causica.functional_relationships import DECIEmbedFunctionalRelationships
from causica.graph.dag_constraint import calculate_dagness
from Preprocess import preprocess, determine_types


def process(df, outcome: str, treatment_names: str, epochs: int, input_colums: str = ""):
    if len(input_colums) > 0:
        try:
            df = df[[col.strip() for col in input_colums.split(',')]]
        except KeyError as e:
            warnings.warn(f"Column not found: {e}")
            df = df[[]]
    df=preprocess(df, drop=False, stratergy='mean', code='Label', scale='Standard')
    metadata=[{'name': col, 'type': determine_types(col), 'group_name': col} for col in df.columns]
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    variables_path = "metadata.json"

    @dataclass(frozen=True)
    class TrainingConfig:
        noise_dist: ContinuousNoiseDist = ContinuousNoiseDist.SPLINE
        batch_size: int = 2048
        max_epoch: int = epochs
        gumbel_temp: float = 0.25
        averaging_period: int = 10
        prior_sparsity_lambda: float = 5.0
        init_rho: float = 1.0
        init_alpha: float = 0.0

    training_config = TrainingConfig()
    auglag_config = AugLagLRConfig()
    seed = 1

    pl.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    columns = list(df.columns)

    tensor_dict_fields = {}
    for i, col in enumerate(columns):
        tensor_dict_fields[col] = torch.tensor(df[col].values, dtype=torch.float32).view(-1, 1)

    tensor_dict = TensorDict(
        source=tensor_dict_fields,
        batch_size=torch.Size([df.shape[0]]),
        device=None,
    )

    dataloader_train = DataLoader(
        dataset=tensor_dict,
        collate_fn=lambda x: x,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
    )

    num_nodes = len(tensor_dict.keys())

    prior = GibbsDAGPrior(num_nodes=num_nodes, sparsity_lambda=training_config.prior_sparsity_lambda)

    adjacency_dist: DistributionModule[AdjacencyDistribution] = ENCOAdjacencyDistributionModule(num_nodes)

    functional_relationships = DECIEmbedFunctionalRelationships(
        shapes=tensordict_shapes(tensor_dict),
        embedding_size=32,
        out_dim_g=32,
        num_layers_g=2,
        num_layers_zeta=2,
    )

    types_dict = {col: VariableTypeEnum.CONTINUOUS for col in columns}

    noise_submodules = create_noise_modules(tensordict_shapes(tensor_dict), types_dict, training_config.noise_dist)
    noise_module = JointNoiseModule(noise_submodules)

    sem_module: SEMDistributionModule = SEMDistributionModule(adjacency_dist, functional_relationships, noise_module)

    sem_module.to(device)

    modules = {
        "functional_relationships": sem_module.functional_relationships,
        "vardist": sem_module.adjacency_module,
        "noise_dist": sem_module.noise_module,
    }
    parameter_list = [
        {"params": module.parameters(), "lr": auglag_config.lr_init_dict[name], "name": name}
        for name, module in modules.items()
    ]

    optimizer = torch.optim.Adam(parameter_list)

    scheduler = AugLagLR(config=auglag_config)
    auglag_loss = AugLagLossCalculator(init_alpha=training_config.init_alpha, init_rho=training_config.init_rho)

    initial_temp = training_config.gumbel_temp
    temperature_decay = 0.95
    num_samples = len(df)

    start_time = time.time()
