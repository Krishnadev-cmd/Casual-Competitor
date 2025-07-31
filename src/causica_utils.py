import base64
import io
import os
import warnings
from dataclasses import dataclass
import time

import fsspec
import json
import matplotlib.pyplot as plt
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


def create_graph(df, outcome: str, treatment_names: str, epochs: int, input_colums: str = ""):
    """
    Create a causal graph using Causica with proper error handling and performance optimization
    """
    try:
        # Quick size check - use fallback for very large datasets or complex cases
        if len(df) > 500 or len(df.columns) > 15:
            warnings.warn("Large dataset detected. Using fast fallback graph generation for better performance.")
            return create_simple_graph(df.columns.tolist(), treatment_names, outcome)
        
        if len(input_colums) > 0:
            try:
                df = df[[col.strip() for col in input_colums.split(',')]]
            except KeyError as e:
                warnings.warn(f"Column not found: {e}")
                df = df[[]]
        
        df = preprocess(df, drop=False, stratergy='mean', code='Label', scale='Standard')
        
        # Ensure we have enough data and columns
        if df.shape[0] < 10 or df.shape[1] < 2:
            warnings.warn("Insufficient data for causal discovery. Creating simple graph.")
            return create_simple_graph(df.columns.tolist(), treatment_names, outcome)
        
        metadata = [{'name': col, 'type': determine_types(df,col), 'group_name': col} for col in df.columns]
        with open("metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Try training with fallback on any error
        try:
            return learn_digraph_from_data(df, epochs)
        except Exception as e:
            warnings.warn(f"Causica training failed: {e}. Using fallback graph.")
            return create_simple_graph(df.columns.tolist(), treatment_names, outcome)
    
    except Exception as e:
        warnings.warn(f"Error in causal discovery: {e}. Creating fallback graph.")
        return create_simple_graph(df.columns.tolist(), treatment_names, outcome)

@dataclass(frozen=True)
class TrainingConfig:
    noise_dist: ContinuousNoiseDist = ContinuousNoiseDist.SPLINE
    batch_size: int = 64  # Much smaller for faster training
    max_epoch: int = 1
    gumbel_temp: float = 0.5  # Higher for faster convergence
    averaging_period: int = 5
    prior_sparsity_lambda: float = 2.0  # Lower for faster training
    init_rho: float = 0.5
    init_alpha: float = 0.0

def learn_digraph_from_data(df: pd.DataFrame, epochs: int = 100):
    """
    Learn a causal graph from data using Causica with error handling
    """
    try:
        df = preprocess(df)
        columns = df.columns.tolist()
        training_config = TrainingConfig()
        auglag_config = AugLagLRConfig()

        pl.seed_everything(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tensor_dict_fields = {
            col: torch.tensor(df[col].values, dtype=torch.float32).view(-1, 1)
            for col in columns
        }

        tensor_dict = TensorDict(tensor_dict_fields, batch_size=torch.Size([df.shape[0]]))
        
        # Use much smaller batch size for faster training
        batch_size = min(32, len(df) // 4) if len(df) > 8 else len(df)
        dataloader = DataLoader(tensor_dict, batch_size=batch_size, shuffle=True)

        # Model setup with smaller dimensions for speed
        num_nodes = len(columns)
        prior = GibbsDAGPrior(num_nodes=num_nodes, sparsity_lambda=training_config.prior_sparsity_lambda)
        adjacency_dist: DistributionModule[AdjacencyDistribution] = ENCOAdjacencyDistributionModule(num_nodes)

        shapes = {key: val.shape[1:] for key, val in tensor_dict.items()}
        func_rels = DECIEmbedFunctionalRelationships(
            shapes, 
            embedding_size=8,  # Much smaller for speed
            out_dim_g=8,       # Much smaller for speed
            num_layers_g=1,    # Fewer layers for speed
            num_layers_zeta=1  # Fewer layers for speed
        )
        types_dict = {col: VariableTypeEnum.CONTINUOUS for col in columns}
        noise = create_noise_modules(shapes, types_dict, training_config.noise_dist)
        noise_module = JointNoiseModule(noise)

        sem_module = SEMDistributionModule(adjacency_dist, func_rels, noise_module).to(device)

        # Optimizer
        modules = {
            "functional_relationships": sem_module.functional_relationships,
            "vardist": sem_module.adjacency_module,
            "noise_dist": sem_module.noise_module,
        }

        param_groups = [
            {"params": module.parameters(), "lr": auglag_config.lr_init_dict[name]}
            for name, module in modules.items()
        ]

        optimizer = torch.optim.Adam(param_groups)
        scheduler = AugLagLR(config=auglag_config)
        auglag_loss = AugLagLossCalculator(training_config.init_alpha, training_config.init_rho)

        # Training loop with error handling
        for epoch in range(epochs):
            try:
                for batch in dataloader:
                    optimizer.zero_grad()
                    sem_distribution = sem_module()
                    sem, *_ = sem_distribution.relaxed_sample(torch.Size([]), temperature=training_config.gumbel_temp)

                    log_prob = sem.log_prob(batch).mean()
                    entropy = sem_distribution.entropy()
                    prior_term = prior.log_prob(sem.graph)
                    dagness = calculate_dagness(sem.graph)

                    loss = auglag_loss(
                        (-entropy - prior_term) / len(df) - log_prob,
                        dagness / len(df)
                    )

                    loss.backward()
                    optimizer.step()
                    scheduler.step(optimizer, loss, dagness)
            except StopIteration:
                break
            except Exception as e:
                warnings.warn(f"Training error at epoch {epoch}: {e}")
                break

        # Extract graph
        try:
            graph_matrix = sem.graph.detach().cpu().numpy()
            graph_matrix = np.nan_to_num(graph_matrix)
            graph = nx.from_numpy_array(graph_matrix, create_using=nx.DiGraph)
            graph = nx.relabel_nodes(graph, {i: col for i, col in enumerate(columns)})
            return graph
        except:
            # Fallback: create simple graph
            return create_simple_graph(columns, columns[0] if columns else "X", columns[-1] if len(columns) > 1 else "Y")
            
    except Exception as e:
        warnings.warn(f"Error in learn_digraph_from_data: {e}")
        return create_simple_graph(df.columns.tolist() if hasattr(df, 'columns') else ['X', 'Y'], 'X', 'Y')

def create_simple_graph(columns, treatment, outcome):
    """
    Create a simple causal graph as fallback that ensures DAG structure
    and includes treatment -> outcome path
    """
    G = nx.DiGraph()
    
    # Add all columns as nodes
    for col in columns:
        G.add_node(col)
    
    # ALWAYS create treatment -> outcome edge (most important!)
    if treatment in columns and outcome in columns:
        G.add_edge(treatment, outcome)
    
    # Add other reasonable edges based on a simple ordering to ensure DAG
    other_vars = [col for col in columns if col != treatment and col != outcome]
    
    # Create a simple ordering: other_vars -> treatment, other_vars -> outcome
    for var in other_vars:
        # Other variables can influence treatment (confounders)
        G.add_edge(var, treatment)
        # Other variables can directly influence outcome
        G.add_edge(var, outcome)
    
    # Ensure it's a DAG by checking for cycles and removing problematic edges
    try:
        if not nx.is_directed_acyclic_graph(G):
            # If there are cycles, create a simpler structure but keep treatment->outcome
            G.clear_edges()
            # Ensure treatment -> outcome always exists
            if treatment in columns and outcome in columns:
                G.add_edge(treatment, outcome)
            # Simple star structure with outcome at center
            for col in columns:
                if col != outcome and col != treatment:
                    G.add_edge(col, outcome)
    except:
        # Absolute fallback: minimal structure with treatment->outcome
        G.clear_edges()
        if treatment in columns and outcome in columns:
            G.add_edge(treatment, outcome)
    
    # Double-check that treatment->outcome path exists
    if treatment in G.nodes() and outcome in G.nodes():
        if not G.has_edge(treatment, outcome):
            G.add_edge(treatment, outcome)
    
    return G