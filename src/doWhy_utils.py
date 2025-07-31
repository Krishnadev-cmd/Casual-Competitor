import pandas as pd
import networkx as nx
import pydot
from dowhy import CausalModel
import matplotlib.pyplot as plt
import os


def digraph_from_nx(G: nx.DiGraph, treatment: str, outcome: str):
    """
    Convert a NetworkX DiGraph to a DOT string usable by DoWhy's CausalModel.
    Ensures the graph is a valid DAG and has treatment->outcome path.
    """
    # Ensure the graph is a DAG
    if not nx.is_directed_acyclic_graph(G):
        # If not a DAG, create a simpler version
        simple_G = nx.DiGraph()
        simple_G.add_nodes_from(G.nodes())
        
        # ALWAYS add treatment -> outcome edge first
        if treatment in G.nodes() and outcome in G.nodes():
            simple_G.add_edge(treatment, outcome)
        
        # Add other edges carefully to maintain DAG property
        for node in G.nodes():
            if node != outcome and node != treatment:
                simple_G.add_edge(node, outcome)  # Other vars -> outcome
                simple_G.add_edge(node, treatment)  # Other vars -> treatment
        
        G = simple_G
    
    # Double-check that treatment->outcome path exists
    if treatment in G.nodes() and outcome in G.nodes():
        if not nx.has_path(G, treatment, outcome):
            # Force add direct edge if no path exists
            G.add_edge(treatment, outcome)
    
    dot_string = "digraph {\n"
    for source, target in G.edges():
        dot_string += f'    "{source}" -> "{target}";\n'
    dot_string += "}"
    
    return dot_string


def run_causal_inference(data: pd.DataFrame, dot_str: str, treatment: str, outcome: str):
    """
    Run DoWhy causal inference using a DOT string graph.
    """
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=dot_str
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    refute = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )

    return model, identified_estimand, estimate, refute


def run_dowhy_inference(model):
    """
    Run DoWhy causal inference using a CausalModel object.
    """
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    refute = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )

    return estimate

def plot_graph(model, path='causal_graph.png'):
    """
    Saves the causal graph plot to file and returns the path.
    """
    model.view_model(layout="dot")
    if os.path.exists("causal_model.png"):
        os.rename("causal_model.png", path)
    return path
