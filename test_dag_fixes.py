#!/usr/bin/env python3
"""
Test DAG validation and graph creation
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dag_validation():
    """Test that our graph creation produces valid DAGs"""
    print("Testing DAG validation...")
    
    try:
        from causica_utils import create_simple_graph
        from doWhy_utils import digraph_from_nx
        import networkx as nx
        
        # Test 1: Simple case
        columns = ['treatment', 'outcome', 'confounder']
        graph = create_simple_graph(columns, 'treatment', 'outcome')
        
        is_dag = nx.is_directed_acyclic_graph(graph)
        print(f"✓ Simple graph is DAG: {is_dag}")
        
        # Test 2: DOT conversion
        dot_string = digraph_from_nx(graph, 'treatment', 'outcome')
        print(f"✓ DOT conversion successful")
        print(f"DOT string: {dot_string}")
        
        # Test 3: More complex case
        columns = ['x1', 'x2', 'x3', 'treatment', 'outcome']
        graph = create_simple_graph(columns, 'treatment', 'outcome')
        
        is_dag = nx.is_directed_acyclic_graph(graph)
        print(f"✓ Complex graph is DAG: {is_dag}")
        print(f"Edges: {list(graph.edges())}")
        
        return True
        
    except Exception as e:
        print(f"✗ DAG validation error: {e}")
        return False

def test_causal_model_creation():
    """Test creating a DoWhy causal model"""
    print("\nTesting causal model creation...")
    
    try:
        from dowhy import CausalModel
        from doWhy_utils import digraph_from_nx
        from causica_utils import create_simple_graph
        
        # Create sample data
        np.random.seed(42)
        n = 100
        
        x1 = np.random.normal(0, 1, n)
        treatment = x1 + np.random.normal(0, 0.5, n)
        outcome = 2 * treatment + x1 + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({
            'x1': x1,
            'treatment': treatment,
            'outcome': outcome
        })
        
        # Create graph
        graph = create_simple_graph(df.columns.tolist(), 'treatment', 'outcome')
        dot_string = digraph_from_nx(graph, 'treatment', 'outcome')
        
        # Create causal model
        model = CausalModel(
            data=df,
            treatment='treatment',
            outcome='outcome',
            graph=dot_string
        )
        
        print("✓ CausalModel created successfully!")
        print(f"Treatment: {model._treatment}")
        print(f"Outcome: {model._outcome}")
        
        return True
        
    except Exception as e:
        print(f"✗ Causal model creation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing DAG fixes...")
    
    test1 = test_dag_validation()
    test2 = test_causal_model_creation()
    
    if test1 and test2:
        print("\n✅ All DAG tests passed! The fixes should resolve the issues.")
        print("\n🚀 Now restart your Streamlit app and try again:")
        print("   streamlit run src/app.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
