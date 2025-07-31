import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import binascii
import base64
import json
import io
import sys
from io import BytesIO
from PIL import Image

from dowhy import CausalModel
from dowhy.utils.graph_operations import str_to_dot
from causica_utils import create_graph
from doWhy_utils import digraph_from_nx, run_dowhy_inference
from Preprocess import preprocess, preprocess_dates
from Utils import remove_id_columns, manupulate_output

# Convert base64 to image
def convert_to_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string, validate=True)
        image_stream = BytesIO(image_data)
        st.image(image_stream, caption="", use_column_width=True)
    except binascii.Error as e:
        st.error(f"Failed to decode base64 string: {e}")

# Traditional analysis

def TraditionalAnalysisApp(df):
    st.header("Dataset Analysis")
    st.write("This section demonstrates traditional statistical analysis.")

    st.write("**Preview of Dataset:**")
    st.write(df.head())

    @st.cache_data
    def get_feature_types(df):
        cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num = df.select_dtypes(include=['number']).columns.tolist()
        dt = df.select_dtypes(include=['datetime64']).columns.tolist()
        return cat, num, dt

    cat_cols, num_cols, dt_cols = get_feature_types(df)

    st.write(f"- Categorical Features: {cat_cols}")
    st.write(f"- Numerical Features: {num_cols}")
    st.write(f"- Datetime Features: {dt_cols}")

    if num_cols:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Main app

def main():
    st.sidebar.title("Causal Discovery")

    display_analysis = st.sidebar.radio("Show dataset analysis?", ("Yes", "No"))
    preprocessed = st.sidebar.radio("Is dataset preprocessed?", ("Yes", "No"))
    fast_mode = st.sidebar.radio("Fast mode?", ("Yes", "No"), help="Skip complex ML training for faster results")
    outcome_variable = st.sidebar.text_input("Outcome variable")
    treatment_variable = st.sidebar.text_input("Treatment variable")
    model_choice = st.sidebar.radio("Choose model", ("causica", "dowhy"))

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

    if st.sidebar.button("Run Analysis") and uploaded_file:
        with st.spinner("Processing data..."):
            df = pd.read_csv(uploaded_file)
            df = remove_id_columns(df)

            if preprocessed == "No":
                df = preprocess(df)
            else:
                df = preprocess_dates(df)

            if display_analysis == "Yes":
                TraditionalAnalysisApp(df)

            df = df.select_dtypes(include=['number'])  # drop non-numeric for simplicity
        
        # Validation
        if not outcome_variable or not treatment_variable:
            st.error("Please specify both outcome and treatment variables.")
            return
            
        if outcome_variable not in df.columns:
            st.error(f"Outcome variable '{outcome_variable}' not found in dataset. Available columns: {list(df.columns)}")
            return
            
        if treatment_variable not in df.columns:
            st.error(f"Treatment variable '{treatment_variable}' not found in dataset. Available columns: {list(df.columns)}")
            return

        st.subheader("Causal Graph")
        if model_choice == "causica" and fast_mode == "No":
            st.info("🧠 Running Causica deep learning model... (this may take a moment)")
            digraph_nx = create_graph(df, outcome_variable, treatment_variable, epochs=1)
            st.write("**Causal Graph (Causica)**")
            dot_string = digraph_from_nx(digraph_nx, treatment_variable, outcome_variable)
            st.graphviz_chart(dot_string)
        elif model_choice == "causica" and fast_mode == "Yes":
            st.info("⚡ Fast mode: Using simple graph structure")
            from causica_utils import create_simple_graph
            digraph_nx = create_simple_graph(df.columns.tolist(), treatment_variable, outcome_variable)
            st.write("**Causal Graph (Fast Mode)**")
            dot_string = digraph_from_nx(digraph_nx, treatment_variable, outcome_variable)
            st.graphviz_chart(dot_string)
        else:  # DoWhy mode
            st.info("📊 Using correlation-based graph discovery")
            # Create a simple graph based on correlations that ensures DAG structure
            corr_matrix = df.corr()
            threshold = 0.3  # correlation threshold
            
            # Create a simple DAG structure
            dot_string = "digraph {\n"
            
            # Always ensure treatment -> outcome edge exists
            dot_string += f'    "{treatment_variable}" -> "{outcome_variable}";\n'
            
            # Add edges from other variables to outcome (not creating cycles)
            for col in df.columns:
                if col != outcome_variable and col != treatment_variable:
                    if abs(corr_matrix.loc[col, outcome_variable]) > threshold:
                        dot_string += f'    "{col}" -> "{outcome_variable}";\n'
            
            # Add edges from other variables to treatment (not creating cycles)
            for col in df.columns:
                if col != outcome_variable and col != treatment_variable:
                    if abs(corr_matrix.loc[col, treatment_variable]) > threshold:
                        dot_string += f'    "{col}" -> "{treatment_variable}";\n'
            
            dot_string += "}"
            
            st.write("**Causal Graph (Correlation-based)**")
            st.graphviz_chart(dot_string)

        # Causal Inference Section

            st.subheader("Causal Inference (DoWhy)")
            try:
                # Validate the DOT string represents a valid DAG
                st.info("Creating causal model...")
                
                # Debug: Show the graph structure
                with st.expander("🔍 Graph Structure Debug"):
                    st.text(f"Treatment: {treatment_variable}")
                    st.text(f"Outcome: {outcome_variable}")
                    st.text("Graph edges:")
                    st.code(dot_string)
                
                model = CausalModel(data=df,
                                    graph=dot_string,
                                    treatment=treatment_variable,
                                    outcome=outcome_variable)
                
                st.success("✅ Causal model created successfully!")
                
                # Save and display graph
                try:
                    fig = plt.figure(figsize=(10, 8))
                    model.view_model(layout="dot")
                    st.pyplot(fig)
                    plt.close()
                except Exception as viz_error:
                    st.warning(f"Advanced graph visualization not available: {str(viz_error)[:100]}...")
                    st.info("💡 Graph structure is still being used for causal inference.")
                
                st.info("Running causal inference...")
                inference_result = run_dowhy_inference(model)
                
                if "zero" in str(inference_result).lower() or "0.0" in str(inference_result):
                    st.warning("⚠️ Causal effect appears to be zero or very small")
                    st.info("This might indicate:")
                    st.info("• No causal relationship in the data")
                    st.info("• Insufficient data or weak signal")
                    st.info("• Graph structure issues")
                else:
                    st.success("✅ Causal inference completed!")
                
                st.markdown(
                    f"""
                    <div style="font-size:16px; font-weight:bold; color:#007BFF;">
                        {manupulate_output(str(inference_result), treatment_variable, outcome_variable, "$")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show raw inference result for debugging
                with st.expander("🔬 Raw Inference Result"):
                    st.write(str(inference_result))
                
            except Exception as model_error:
                error_msg = str(model_error)
                if "acyclic" in error_msg.lower():
                    st.error("❌ Graph contains cycles (not a valid DAG)")
                    st.info("💡 Try using the 'dowhy' model option which creates simpler graphs")
                    st.info("🔧 Or check your data for logical causal relationships")
                elif "treatment" in error_msg.lower() or "outcome" in error_msg.lower():
                    st.error(f"❌ Variable not found: {error_msg}")
                    st.info(f"Available columns: {list(df.columns)}")
                else:
                    st.error(f"❌ Error creating causal model: {error_msg}")
                    st.info("💡 This might be due to data format or graph structure issues.")
                
                # Show debug information
                st.expander("🔍 Debug Information").write({
                    "Error": str(model_error),
                    "Treatment": treatment_variable,
                    "Outcome": outcome_variable,
                    "Columns": list(df.columns),
                    "Graph": dot_string[:200] + "..." if len(dot_string) > 200 else dot_string
                })

        else:  # dowhy correlation-based structure
            # Create a simple graph based on correlations that ensures DAG structure
            corr_matrix = df.corr()
            threshold = 0.3  # correlation threshold
            
            # Create a simple DAG structure
            dot_string = "digraph {\n"
            
            # Always ensure treatment -> outcome edge exists
            dot_string += f'    "{treatment_variable}" -> "{outcome_variable}";\n'
            
            # Add edges from other variables to outcome (not creating cycles)
            for col in df.columns:
                if col != outcome_variable and col != treatment_variable:
                    if abs(corr_matrix.loc[col, outcome_variable]) > threshold:
                        dot_string += f'    "{col}" -> "{outcome_variable}";\n'
            
            # Add edges from other variables to treatment (not creating cycles)
            for col in df.columns:
                if col != outcome_variable and col != treatment_variable:
                    if abs(corr_matrix.loc[col, treatment_variable]) > threshold:
                        dot_string += f'    "{col}" -> "{treatment_variable}";\n'
            
            dot_string += "}"
            
            st.graphviz_chart(dot_string)
            
            try:
                st.info("Creating causal model with correlation-based graph...")
                
                model = CausalModel(data=df,
                                    graph=dot_string,
                                    treatment=treatment_variable,
                                    outcome=outcome_variable)
                
                st.success("✅ Causal model created successfully!")
                
                try:
                    fig = plt.figure(figsize=(10, 8))
                    model.view_model(layout="dot")
                    st.pyplot(fig)
                    plt.close()
                except Exception as viz_error:
                    st.warning(f"Advanced graph visualization not available: {str(viz_error)[:100]}...")
                    
                st.info("Running causal inference...")
                inference_result = run_dowhy_inference(model)
                
                st.success("✅ Causal inference completed!")
                st.markdown(
                    f"""
                    <div style="font-size:16px; font-weight:bold; color:#007BFF;">
                        {manupulate_output(str(inference_result), treatment_variable, outcome_variable, "$")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as model_error:
                error_msg = str(model_error)
                if "acyclic" in error_msg.lower():
                    st.error("❌ Graph contains cycles (not a valid DAG)")
                    st.info("💡 The correlation-based approach created cycles. Try the 'causica' option instead.")
                elif "treatment" in error_msg.lower() or "outcome" in error_msg.lower():
                    st.error(f"❌ Variable not found: {error_msg}")
                    st.info(f"Available columns: {list(df.columns)}")
                else:
                    st.error(f"❌ Error creating causal model: {error_msg}")
                    
                # Show debug information
                st.expander("🔍 Debug Information").write({
                    "Error": str(model_error),
                    "Treatment": treatment_variable,
                    "Outcome": outcome_variable,
                    "Columns": list(df.columns),
                    "Graph": dot_string[:200] + "..." if len(dot_string) > 200 else dot_string
                })

if __name__ == "__main__":
    main()