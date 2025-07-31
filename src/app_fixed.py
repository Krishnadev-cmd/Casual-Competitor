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
from causica_utils import create_graph, create_simple_graph
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
    st.title("🧠 Causal Discovery & Inference")
    st.markdown("Discover causal relationships in your data using advanced ML techniques.")
    
    st.sidebar.title("⚙️ Configuration")

    display_analysis = st.sidebar.radio("Show dataset analysis?", ("Yes", "No"))
    preprocessed = st.sidebar.radio("Is dataset preprocessed?", ("Yes", "No"))
    fast_mode = st.sidebar.radio("⚡ Fast mode?", ("Yes", "No"), 
                                help="Skip complex ML training for faster results")
    
    st.sidebar.markdown("---")
    outcome_variable = st.sidebar.text_input("🎯 Outcome variable", 
                                           placeholder="e.g., sales, profit")
    treatment_variable = st.sidebar.text_input("💊 Treatment variable", 
                                             placeholder="e.g., price, marketing")
    
    model_choice = st.sidebar.radio("🤖 Choose model", ("causica", "dowhy"),
                                  help="Causica: Deep learning, DoWhy: Correlation-based")

    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type="csv")

    if st.sidebar.button("🚀 Run Analysis", type="primary") and uploaded_file:
        with st.spinner("Processing data..."):
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
                
                df = remove_id_columns(df)
                
                if preprocessed == "No":
                    df = preprocess(df)
                else:
                    df = preprocess_dates(df)

                if display_analysis == "Yes":
                    TraditionalAnalysisApp(df)

                df = df.select_dtypes(include=['number'])  # drop non-numeric for simplicity
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                return
        
        # Validation
        if not outcome_variable or not treatment_variable:
            st.error("❌ Please specify both outcome and treatment variables.")
            return
            
        if outcome_variable not in df.columns:
            st.error(f"❌ Outcome variable '{outcome_variable}' not found in dataset.")
            st.info(f"Available columns: {list(df.columns)}")
            return
            
        if treatment_variable not in df.columns:
            st.error(f"❌ Treatment variable '{treatment_variable}' not found in dataset.")
            st.info(f"Available columns: {list(df.columns)}")
            return

        # Causal Graph Generation
        st.subheader("🕸️ Causal Graph")
        
        with st.spinner("Generating causal graph..."):
            if model_choice == "causica" and fast_mode == "No":
                st.info("🧠 Running Causica deep learning model... (this may take a moment)")
                digraph_nx = create_graph(df, outcome_variable, treatment_variable, epochs=1)
                st.write("**Causal Graph (Causica)**")
                dot_string = digraph_from_nx(digraph_nx, treatment_variable, outcome_variable)
                
            elif model_choice == "causica" and fast_mode == "Yes":
                st.info("⚡ Fast mode: Using simple graph structure")
                digraph_nx = create_simple_graph(df.columns.tolist(), treatment_variable, outcome_variable)
                st.write("**Causal Graph (Fast Mode)**")
                dot_string = digraph_from_nx(digraph_nx, treatment_variable, outcome_variable)
                
            else:  # DoWhy correlation-based
                st.info("📊 Using correlation-based graph discovery")
                corr_matrix = df.corr()
                threshold = 0.3
                
                dot_string = "digraph {\n"
                dot_string += f'    "{treatment_variable}" -> "{outcome_variable}";\n'
                
                for col in df.columns:
                    if col != outcome_variable and col != treatment_variable:
                        if abs(corr_matrix.loc[col, outcome_variable]) > threshold:
                            dot_string += f'    "{col}" -> "{outcome_variable}";\n'
                        if abs(corr_matrix.loc[col, treatment_variable]) > threshold:
                            dot_string += f'    "{col}" -> "{treatment_variable}";\n'
                
                dot_string += "}"
                st.write("**Causal Graph (Correlation-based)**")
            
            # Display graph
            st.graphviz_chart(dot_string)

        # Causal Inference
        st.subheader("🔍 Causal Inference")
        
        with st.spinner("Running causal inference..."):
            try:
                # Debug info
                with st.expander("🔍 Graph Structure Debug"):
                    st.text(f"Treatment: {treatment_variable}")
                    st.text(f"Outcome: {outcome_variable}")
                    st.text("Graph edges:")
                    st.code(dot_string)
                
                # Create causal model
                model = CausalModel(data=df,
                                  graph=dot_string,
                                  treatment=treatment_variable,
                                  outcome=outcome_variable)
                
                st.success("✅ Causal model created successfully!")
                
                # Visualization
                try:
                    fig = plt.figure(figsize=(10, 8))
                    model.view_model(layout="dot")
                    st.pyplot(fig)
                    plt.close()
                except Exception as viz_error:
                    st.warning("⚠️ Advanced graph visualization not available")
                    st.info("💡 Graph structure is still being used for causal inference.")
                
                # Run inference
                inference_result = run_dowhy_inference(model)
                
                # Results
                if "zero" in str(inference_result).lower() or "0.0" in str(inference_result):
                    st.warning("⚠️ Causal effect appears to be zero or very small")
                    st.info("This might indicate:")
                    st.info("• No causal relationship in the data")
                    st.info("• Insufficient data or weak signal")
                    st.info("• Try different treatment/outcome variables")
                else:
                    st.success("✅ Causal inference completed!")
                
                # Display results
                st.markdown("### 📊 Results")
                st.markdown(
                    f"""
                    <div style="font-size:16px; font-weight:bold; color:#007BFF; 
                               background-color:#f0f8ff; padding:15px; border-radius:10px; 
                               border-left:5px solid #007BFF;">
                        {manupulate_output(str(inference_result), treatment_variable, outcome_variable, "$")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Raw results for debugging
                with st.expander("🔬 Raw Inference Result"):
                    st.write(str(inference_result))
                
            except Exception as model_error:
                error_msg = str(model_error)
                st.error("❌ Error in causal inference")
                
                if "acyclic" in error_msg.lower():
                    st.info("💡 Graph contains cycles. Try fast mode or DoWhy option.")
                elif "treatment" in error_msg.lower() or "outcome" in error_msg.lower():
                    st.info(f"💡 Variable not found. Available: {list(df.columns)}")
                else:
                    st.info("💡 Check your data format and variable names.")
                
                with st.expander("🔍 Error Details"):
                    st.write({
                        "Error": str(model_error),
                        "Treatment": treatment_variable,
                        "Outcome": outcome_variable,
                        "Available Columns": list(df.columns)
                    })

    elif not uploaded_file:
        st.info("👆 Please upload a CSV file to get started!")
        st.markdown("""
        ### 📋 Instructions:
        1. **Upload** your CSV dataset
        2. **Specify** treatment and outcome variables
        3. **Choose** analysis method (Causica for advanced ML, DoWhy for correlations)
        4. **Enable fast mode** for quicker results on large datasets
        5. **Run analysis** to discover causal relationships!
        
        ### 💡 Tips:
        - **Treatment**: The variable you want to change (e.g., price, marketing spend)
        - **Outcome**: The variable you want to predict (e.g., sales, profit)
        - **Fast mode**: Recommended for datasets with >500 rows or >15 columns
        """)

if __name__ == "__main__":
    main()
