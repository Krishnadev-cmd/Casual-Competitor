#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from doWhy_utils import digraph_from_nx, run_dowhy_inference
        from Preprocess import preprocess
        from Utils import remove_id_columns, manupulate_output
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality"""
    print("\nTesting preprocessing...")
    try:
        from Preprocess import preprocess
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'target': np.random.normal(10, 3, 100)
        })
        
        processed_df = preprocess(df)
        print(f"✓ Preprocessing successful. Original shape: {df.shape}, Processed shape: {processed_df.shape}")
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    try:
        from Utils import remove_id_columns, manupulate_output
        
        # Test remove_id_columns
        df = pd.DataFrame({
            'id': range(100),
            'user_id': range(100, 200),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
        })
        
        cleaned_df = remove_id_columns(df)
        print(f"✓ ID removal successful. Original columns: {list(df.columns)}, Cleaned columns: {list(cleaned_df.columns)}")
        
        # Test manupulate_output
        sample_result = "The causal effect estimate is 2.5 with confidence interval [1.2, 3.8]"
        output = manupulate_output(sample_result, "treatment", "outcome", "$")
        print(f"✓ Output manipulation successful")
        
        return True
    except Exception as e:
        print(f"✗ Utils test error: {e}")
        return False

def test_dowhy_functions():
    """Test DoWhy utility functions"""
    print("\nTesting DoWhy functions...")
    try:
        from doWhy_utils import digraph_from_nx
        import networkx as nx
        
        # Create a simple graph
        G = nx.DiGraph()
        G.add_edges_from([('X', 'Y'), ('X', 'Z'), ('Y', 'Z')])
        
        dot_string = digraph_from_nx(G, treatment='X', outcome='Z')
        print(f"✓ DiGraph to DOT conversion successful")
        print(f"DOT string: {dot_string}")
        
        return True
    except Exception as e:
        print(f"✗ DoWhy functions error: {e}")
        return False

def test_streamlit_app_syntax():
    """Test that the Streamlit app can be parsed (syntax check)"""
    print("\nTesting Streamlit app syntax...")
    try:
        with open('src/app.py', 'r') as f:
            content = f.read()
        
        # Try to compile the code to check for syntax errors
        compile(content, 'src/app.py', 'exec')
        print("✓ Streamlit app syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Streamlit app syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Streamlit app test error: {e}")
        return False

if __name__ == "__main__":
    print("Running simplified tests for causal inference pipeline...")
    
    tests = [
        test_basic_imports,
        test_preprocessing,
        test_utils,
        test_dowhy_functions,
        test_streamlit_app_syntax
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic tests passed! Your core functionality is working.")
        print("\nNext steps:")
        print("1. Install any missing dependencies if needed")
        print("2. Test the Streamlit app: streamlit run src/app.py")
        print("3. For full causal discovery testing, you may need to adjust Causica parameters")
    else:
        print("✗ Some tests failed. Check the error messages above.")
