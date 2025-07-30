"""
Causal Inference Pipeline using DECI (Deep End-to-end Causal Inference)

This module implements a complete causal discovery pipeline that:
1. Preprocesses data for causal analysis
2. Learns causal relationships between variables using deep learning
3. Constructs a Directed Acyclic Graph (DAG) representing causal structure
4. Exports the discovered causal graph in various formats

The DECI model uses:
- Structural Equation Models (SEMs) for causal relationships
- Variational inference for learning graph structure
- Augmented Lagrangian optimization for DAG constraints
- Deep neural networks for functional relationships
"""

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

# Causica imports for causal discovery
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
    """
    Main function for causal discovery using DECI (Deep End-to-end Causal Inference)
    
    This function performs the complete causal discovery pipeline:
    1. Data preprocessing and feature selection
    2. Model configuration and setup
    3. Training the DECI model to learn causal relationships
    4. Extracting and visualizing the causal graph
    5. Fine-tuning with constraints and exporting results
    
    Args:
        df (pd.DataFrame): Input dataset for causal analysis
        outcome (str): Target variable (outcome of interest)
        treatment_names (str): Treatment variables (comma-separated if multiple)
        epochs (int): Number of training epochs for the model
        input_colums (str, optional): Specific columns to use (comma-separated). 
                                    If empty, uses all columns.
    
    Returns:
        Trained DECI model and saves causal graph to files
    """
    
    # Step 1: Data Selection and Preprocessing
    # =======================================
    if len(input_colums) > 0:
        try:
            # Select only specified columns if provided
            df = df[[col.strip() for col in input_colums.split(',')]]
        except KeyError as e:
            warnings.warn(f"Column not found: {e}")
            df = df[[]]
    
    # Preprocess the data: handle missing values, encode categoricals, scale features
    df = preprocess(df, drop=False, stratergy='mean', code='Label', scale='Standard')
    
    # Create metadata for each variable (required by Causica framework)
    metadata = [{'name': col, 'type': determine_types(df,col), 'group_name': col} for col in df.columns]
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    variables_path = "metadata.json"

    # Step 2: Training Configuration Setup
    # ===================================
    @dataclass(frozen=True)
    class TrainingConfig:
        """Configuration parameters for DECI model training"""
        noise_dist: ContinuousNoiseDist = ContinuousNoiseDist.SPLINE  # Noise distribution type
        batch_size: int = 2048                    # Batch size for training
        max_epoch: int = epochs                   # Maximum training epochs
        gumbel_temp: float = 0.25                # Gumbel softmax temperature for discrete sampling
        averaging_period: int = 10                # Period for averaging metrics
        prior_sparsity_lambda: float = 5.0       # Sparsity regularization strength
        init_rho: float = 1.0                    # Initial penalty parameter for DAG constraint
        init_alpha: float = 0.0                  # Initial Lagrange multiplier

    training_config = TrainingConfig()
    auglag_config = AugLagLRConfig()  # Augmented Lagrangian learning rate configuration
    seed = 1

    # Set random seeds for reproducibility
    pl.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 3: Data Preparation for PyTorch
    # ====================================
    columns = list(df.columns)

    # Convert pandas DataFrame to TensorDict format (required by Causica)
    tensor_dict_fields = {}
    for i, col in enumerate(columns):
        tensor_dict_fields[col] = torch.tensor(df[col].values, dtype=torch.float32).view(-1, 1)

    tensor_dict = TensorDict(
        source=tensor_dict_fields,
        batch_size=torch.Size([df.shape[0]]),
        device=None,
    )

    # Create DataLoader for batch processing during training
    dataloader_train = DataLoader(
        dataset=tensor_dict,
        collate_fn=lambda x: x,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
    )

    # Step 4: Model Architecture Setup
    # ===============================
    num_nodes = len(tensor_dict.keys())

    # Prior distribution over DAG structures (encourages sparsity)
    prior = GibbsDAGPrior(num_nodes=num_nodes, sparsity_lambda=training_config.prior_sparsity_lambda)

    # Adjacency matrix distribution (learns which edges exist in the causal graph)
    adjacency_dist: DistributionModule[AdjacencyDistribution] = ENCOAdjacencyDistributionModule(num_nodes)

    # Functional relationships module (learns how variables causally affect each other)
    functional_relationships = DECIEmbedFunctionalRelationships(
        shapes=tensordict_shapes(tensor_dict),
        embedding_size=32,          # Embedding dimension for variables
        out_dim_g=32,              # Output dimension for g functions
        num_layers_g=2,            # Number of layers in g functions
        num_layers_zeta=2,         # Number of layers in zeta functions
    )

    # Define variable types (all continuous for now)
    types_dict = {col: VariableTypeEnum.CONTINUOUS for col in columns}

    # Noise modules for each variable (models unexplained variance)
    noise_submodules = create_noise_modules(tensordict_shapes(tensor_dict), types_dict, training_config.noise_dist)
    noise_module = JointNoiseModule(noise_submodules)

    # Complete SEM (Structural Equation Model) combining all components
    sem_module: SEMDistributionModule = SEMDistributionModule(adjacency_dist, functional_relationships, noise_module)

    sem_module.to(device)

    # Step 5: Optimizer and Training Setup
    # ===================================
    # Group model components with different learning rates
    modules = {
        "functional_relationships": sem_module.functional_relationships,
        "vardist": sem_module.adjacency_module,
        "noise_dist": sem_module.noise_module,
    }
    
    # Create parameter groups with component-specific learning rates
    parameter_list = [
        {"params": module.parameters(), "lr": auglag_config.lr_init_dict[name], "name": name}
        for name, module in modules.items()
    ]

    optimizer = torch.optim.Adam(parameter_list)

    # Augmented Lagrangian learning rate scheduler (handles DAG constraints)
    scheduler = AugLagLR(config=auglag_config)
    auglag_loss = AugLagLossCalculator(init_alpha=training_config.init_alpha, init_rho=training_config.init_rho)

    # Temperature annealing for Gumbel softmax (starts high, decreases over time)
    initial_temp = training_config.gumbel_temp
    temperature_decay = 0.95
    num_samples = len(df)

    # Step 6: Main Training Loop
    # =========================
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Anneal temperature for Gumbel softmax (makes sampling more discrete over time)
        current_temp = initial_temp * (temperature_decay ** epoch)
        
        for i, batch in enumerate(dataloader_train):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            # Sample a SEM from the distribution
            sem_distribution = sem_module()

            # Use relaxed sampling with current temperature
            sem, *_ = sem_distribution.relaxed_sample(
                torch.Size([]), temperature=current_temp
            )

            # Calculate loss components:
            batch_log_prob = sem.log_prob(batch).mean()          # Data likelihood
            sem_distribution_entropy = sem_distribution.entropy() # Entropy regularization
            prior_term = prior.log_prob(sem.graph)               # Prior over graph structure
            
            # Combined objective (negative ELBO)
            objective = (-sem_distribution_entropy - prior_term) / num_samples - batch_log_prob
            
            # DAG constraint (penalizes cycles in the graph)
            constraint = calculate_dagness(sem.graph)

            # Augmented Lagrangian loss (handles constrained optimization)
            loss = auglag_loss(objective, constraint / num_samples)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            # Update learning rates and penalties
            scheduler.step(
                optimizer=optimizer,
                loss=auglag_loss,
                loss_value=loss,
                lagrangian_penalty=constraint,
            )

            # Progress tracking and logging
            batch_time = time.time() - batch_start_time
            batches_remaining = len(dataloader_train) - (i + 1)
            epoch_time_remaining = batches_remaining * batch_time

            if i % 10 == 0:
                print(
                    f"epoch:{epoch} batch:{i} loss:{loss.item():.5g} nll:{-batch_log_prob.detach().cpu().numpy():.5g} "
                    f"dagness:{constraint.item():.5f} num_edges:{(sem.graph > 0.0).sum()} "
                    f"alpha:{auglag_loss.alpha:.5g} rho:{auglag_loss.rho:.5g} "
                    f"step:{scheduler.outer_opt_counter}|{scheduler.step_counter} "
                    f"num_lr_updates:{scheduler.num_lr_updates}"
                )

        # Epoch completion and time estimation
        epoch_time = time.time() - epoch_start_time
        epochs_remaining = epochs - (epoch + 1)
        total_time_remaining = epochs_remaining * epoch_time
        st.session_state.total_time_remaining = total_time_remaining

        print(
            f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. "
            f"Estimated time remaining: {total_time_remaining:.2f}s"
        )

    # Step 7: Extract and Visualize Causal Graph
    # ==========================================
    # Get the learned adjacency distribution and extract the most likely graph
    vardist = adjacency_dist()
    labels = {col: idx for idx, col in enumerate(df.columns)}
    numpy_array = vardist.mode.cpu().numpy()  # Convert to numpy for visualization
    numpy_array = numpy_array.astype(np.float32)
    numpy_array = np.nan_to_num(numpy_array)  # Handle any NaN values

    # Create NetworkX graph for visualization
    graph = nx.from_numpy_array(numpy_array, create_using=nx.DiGraph)
    nx.draw_networkx(graph, with_labels=True, arrows=True)

    # Export graph in GML format for external analysis
    with open("fin_graph.txt", "w") as f:
        f.write("graph [\n")
        f.write("  directed 1\n")

        # Write nodes with labels
        for label, node_id in labels.items():
            f.write(f"  node [\n    id {node_id}\n    label \"{label}\"\n  ]\n")

        # Write edges
        for source, target in graph.edges():
            f.write(f"  edge [\n    source {source}\n    target {target}\n  ]\n")

        f.write("]\n")

    # Step 8: Fine-tuning with PyTorch Lightning
    # ==========================================
    # Load variable specifications for Lightning module
    with fsspec.open(variables_path, mode="r", encoding="utf-8") as f:
        variables_spec = json.load(f)

    pprint(variables_spec)

    # Create PyTorch Lightning data module with normalization
    data_module = BasicDECIDataModule(
        df,
        variables=[Variable.from_dict(d) for d in variables_spec],
        batch_size=128,
        normalize=True,  # Normalize data for better training stability
    )
    
    print(type(data_module.dataset_train))
    print(data_module.dataset_train.keys())
    num_nodes = len(data_module.dataset_train.keys())

    # Step 9: Set Up Causal Constraints
    # =================================
    # Create constraint matrix to encode domain knowledge
    node_name_to_idx = {key: i for i, key in enumerate(data_module.dataset_train.keys())}
    constraint_matrix = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)

    # Constraint: outcome variable cannot cause other variables (only be caused by them)
    revenue_idx = node_name_to_idx[outcome]
    constraint_matrix[revenue_idx, :] = 0.0  # No outgoing edges from outcome

    # Set random seed for reproducible Lightning training
    pl.seed_everything(seed=1)

    # Step 10: Configure Lightning Module for Final Training
    # ======================================================
    lightning_module = DECIModule(
        noise_dist=ContinuousNoiseDist.GAUSSIAN,  # Use Gaussian noise distribution
        prior_sparsity_lambda=43.0,               # Strong sparsity prior (encourages fewer edges)
        init_rho=30.0,                           # Higher penalty for DAG constraint violations
        init_alpha=0.20,                         # Initial Lagrange multiplier
        auglag_config=AugLagLRConfig(
            max_inner_steps=3400,                # Maximum inner optimization steps
            max_outer_steps=8,                   # Maximum outer optimization steps
            lr_init_dict={                       # Component-specific learning rates
                "icgnn": 0.00076,               # ICGNN learning rate
                "vardist": 0.0098,              # Variational distribution learning rate
                "functional_relationships": 3e-4, # Functional relationships learning rate
                "noise_dist": 0.0070,           # Noise distribution learning rate
            },
        ),
    )

    # Apply the constraint matrix to the Lightning module
    lightning_module.constraint_matrix = torch.tensor(constraint_matrix)

    # Configure PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="auto",                       # Use GPU if available, otherwise CPU
        max_epochs=2000,                         # Maximum number of epochs
        callbacks=[TQDMProgressBar(refresh_rate=19)],  # Progress bar for monitoring
        enable_checkpointing=False,              # Disable checkpointing for simplicity
    )

    # Train the model using Lightning's fit method
    trainer.fit(lightning_module, datamodule=data_module)

    # Step 11: Save and Load Final Model
    # ==================================
    # Save the trained SEM module for later use
    torch.save(lightning_module.sem_module, "deci.pt")

    # Load the saved model and extract the final causal graph
    sem_module: SEMDistributionModule = torch.load("deci.pt")
    sem = sem_module().mode  # Get the most likely SEM configuration

    # Step 12: Create Final Graph Visualization
    # ========================================
    # Convert the learned adjacency matrix to NetworkX graph with proper node labels
    graph = nx.from_numpy_array(sem.graph.cpu().numpy(), create_using=nx.DiGraph)
    graph = nx.relabel_nodes(graph, dict(enumerate(data_module.dataset_train.keys())))

    # Load the previously saved graph structure for layout consistency
    with fsspec.open("fin_graph.txt", mode="r", encoding="utf-8") as f:
        true_adj = nx.parse_gml(f.read())

    # Try to use Graphviz for better layout, fall back to spring layout if unavailable
    try:
        layout = nx.nx_agraph.graphviz_layout(true_adj, prog="dot")
    except (ModuleNotFoundError, ImportError):
        layout = nx.layout.spring_layout(true_adj)

    # Create the final graph visualization
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))

    # Plot nodes with labels
    for node, i in labels.items():
        axis.scatter(layout[node][0], layout[node][1], label=f"{i}: {node}")
    axis.legend()

    # Draw the complete network graph
    nx.draw_networkx(graph, pos=layout, with_labels=True, arrows=True, labels=labels, ax=axis)

    # Step 13: Causal Effect Estimation (Average Treatment Effects)
    # ============================================================
    revenue_estimated_ate = {}  # Store ATE results for each treatment
    num_samples = 2000          # Number of samples for Monte Carlo estimation
    sample_shape = torch.Size([num_samples])
    normalizer = data_module.normalizer  # Get the data normalizer for proper scaling

    # Parse treatment variables from input string
    treatment_columns = [col.strip() for col in treatment_names.split(", ")]

    # Calculate ATE for each treatment variable
    for treatment in treatment_columns:
        ate_means = []  # Store mean effects
        ate_stds = []   # Store standard deviations
        
        # Create intervention values across the range of the treatment variable
        treatment_values = torch.linspace(0.0, df[treatment].max(), steps=11)

        # For each intervention value, estimate the causal effect
        for value in treatment_values:
            # Create intervention: set treatment variable to specific value
            intervention = TensorDict({treatment: torch.tensor([value])}, batch_size=tuple())

            # Sample outcomes under the intervention using do-calculus
            outcome_samples = normalizer.inv(sem.do(interventions=intervention).sample(sample_shape))[outcome]

            # Calculate statistics of the outcome distribution
            mean_value = outcome_samples.mean(0).cpu().numpy()[0]
            std_value = outcome_samples.std(0).cpu().numpy()[0]

            ate_means.append(mean_value)
            ate_stds.append(std_value)

        # Store results for this treatment
        revenue_estimated_ate[treatment] = {
            "ate_means": np.array(ate_means),
            "ate_stds": np.array(ate_stds),
            "treatment_values": treatment_values.cpu().numpy()
        }

    # Convert numpy arrays to lists for JSON serialization
    for treatment in revenue_estimated_ate:
        revenue_estimated_ate[treatment]["ate_means"] = revenue_estimated_ate[treatment]["ate_means"].tolist()
        revenue_estimated_ate[treatment]["ate_stds"] = revenue_estimated_ate[treatment]["ate_stds"].tolist()
        revenue_estimated_ate[treatment]["treatment_values"] = revenue_estimated_ate[treatment][
            "treatment_values"].tolist()

    # Step 14: Save Main Graph Visualization
    # ======================================
    # Convert the main causal graph to base64 for web display
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    graph_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Step 15: Generate Individual Treatment Effect Visualizations
    # ==========================================================
    impact_factor_graphs = {}  # Store individual treatment effect graphs

    # Create a separate graph for each treatment variable
    for i in treatment_columns:
        if i not in revenue_estimated_ate:
            continue  # Skip if no ATE data available

        # Extract ATE results for this treatment variable
        result = revenue_estimated_ate[i]

        # Create month labels for x-axis (placeholder - could be customized)
        months = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]

        # Create error bar plot showing treatment effects over time/values
        plt.figure(figsize=(10, 6))
        plt.errorbar(months, result['ate_means'],
                     yerr=result['ate_stds'], fmt='o-', capsize=5, color='blue')
        plt.title(f'Causal Impact of {i} on {outcome} (Monthly)')
        plt.xlabel('Month')
        plt.ylabel('Average Treatment Effect (ATE)', color='blue')
        plt.grid()
        plt.tight_layout()

        # Convert individual graph to base64 for storage
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_base_64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Store the graph for this treatment variable
        impact_factor_graphs[i] = graph_base_64
        buffer.close()
        plt.close()

    # Return all results: ATE estimates, main graph, and individual treatment graphs
    return revenue_estimated_ate, graph_base64, impact_factor_graphs


def get_time_remaining():
    """
    Utility function to display remaining training time in Streamlit interface
    
    This function checks if there's a time estimate stored in the Streamlit session state
    and displays it to the user. Used for providing real-time feedback during long
    training processes.
    
    Returns:
        float: Estimated remaining time in seconds, or 0 if not available
    """
    with st.empty():
        if 'total_time_remaining' in st.session_state:
            st.write(f"Total Time Remaining: {st.session_state.total_time_remaining:.2f} seconds")
        else:
            st.write("Processing in progress...")
    return st.session_state.get('total_time_remaining', 0)