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


def process(df,input_colums):
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