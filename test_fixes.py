#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Test imports
try:
    from causica_utils import create_graph
    from doWhy_utils import digraph_from_nx, run_dowhy_inference, run_causal_inference
    from Preprocess import preprocess, preprocess_dates
    from Utils import remove_id_columns, manupulate_output
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Create sample data for testing
def create_sample_data():
    np.random.seed(42)
    n = 100
    
    # Generate synthetic data with causal relationships
    x1 = np.random.normal(0, 1, n)
    x2 = 2 * x1 + np.random.normal(0, 0.5, n)  # x2 depends on x1
    treatment = x1 + np.random.normal(0, 0.5, n)  # treatment depends on x1
    outcome = 3 * treatment + x2 + np.random.normal(0, 0.5, n)  # outcome depends on treatment and x2
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'treatment': treatment,
        'outcome': outcome
    })
    
    return df

def test_preprocessing():
    print("\nTesting preprocessing...")
    df = create_sample_data()
    
    try:
        processed_df = preprocess(df)
        print(f"✓ Preprocessing successful. Shape: {processed_df.shape}")
        return processed_df
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return None

def test_causal_graph_creation():
    print("\nTesting causal graph creation...")
    df = create_sample_data()
    
    try:
        # Test with small number of epochs for quick testing
        graph = create_graph(df, outcome='outcome', treatment_names='treatment', epochs=1)
        print(f"✓ Causal graph creation successful. Nodes: {list(graph.nodes())}")
        return graph
    except Exception as e:
        print(f"✗ Causal graph creation error: {e}")
        return None

def test_digraph_conversion():
    print("\nTesting DiGraph to DOT conversion...")
    df = create_sample_data()
    
    try:
        graph = create_graph(df, outcome='outcome', treatment_names='treatment', epochs=1)
        dot_string = digraph_from_nx(graph, treatment='treatment', outcome='outcome')
        print(f"✓ DiGraph conversion successful")
        print(f"DOT string preview: {dot_string[:100]}...")
        return dot_string
    except Exception as e:
        print(f"✗ DiGraph conversion error: {e}")
        return None

if __name__ == "__main__":
    print("Running tests for causal inference pipeline...")
    
    # Run tests
    processed_df = test_preprocessing()
    graph = test_causal_graph_creation()
    dot_string = test_digraph_conversion()
    
    if all([processed_df is not None, graph is not None, dot_string is not None]):
        print("\n✓ All tests passed! The fixes appear to be working correctly.")
    else:
        print("\n✗ Some tests failed. Check the error messages above.")
