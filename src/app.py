import streamlit as st
import pandas as pd
import seaborn as sns
from dowhy import CausalModel
import matplotlib.pyplot as plt
import pandas as pd
import binascii
import json
from io import BytesIO
import io
from Utils import manupulate_output, remove_id_columns
from Preprocess import *
import sys
import base64
from PIL import Image
from io import BytesIO
from causica_utils import process


# Configurations for the plots
# st.set_option('deprecation.showPyplotGlobalUse', False)

# base64 to image
def convert_to_image(base64_string):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string, validate=True)

        # Convert the bytes to an image-like format for Streamlit
        image_stream = BytesIO(image_data)

        # Display the image in the Streamlit app
        st.image(image_stream, caption="", use_column_width=True)

    except binascii.Error as e:
        st.error(f"Failed to decode base64 string: {e}")


def get_digraph(dt, outcome='ProductCost'):
    # Convert the dictionary to a pandas DataFrame

    # Calculate the correlation matrix
    correlation_matrix = dt.corr()

    # Define a threshold for strong correlations
    threshold = 0.1  # Adjust based on what you consider as a strong correlation
    causal_graph = []
    added_edges = set()  # To track added relationships

    # Iterate over the correlation matrix to find pairs with correlation above the threshold
    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if (
                col1 != outcome and col1 != col2
                and abs(correlation_matrix.loc[col1, col2]) >= threshold
                and (col2, col1) not in added_edges  # Check for reverse edge
            ):
                # Add the edge (col1 -> col2) if correlation is above the threshold
                causal_graph.append(f"    {col1} -> {col2};")
                added_edges.add((col1, col2))  # Track this edge

    # Combine the edges to form the final causal graph in dot format
    causal_graph_str = "digraph {\n\n" + "\n".join(causal_graph) + "\n\n}"

    # Return the adjacency list in dot format
    st.write(causal_graph_str)
    return causal_graph_str


def causal_graph(digraph, df, outcome, treatment):
    model = CausalModel(data=df,  # data columns
                        graph=digraph,  # DAG
                        treatment=treatment,  # cause of interest, X
                        outcome=outcome)  # outcome, Y

    st.pyplot(model.view_model())
    return model


def causica_inference(df, outcome: str, treatment_names: str, epochs: int, input_columns: str = ""):
    ate, graph,impact_graph = process(df, outcome, treatment_names, epochs)
    convert_to_image(graph)
    # st.header("Impact Graph")
    # convert first to json
    # json_compatible_string = impact_graph.replace("'", '"')
# Convert to a Python dictionary (JSON object)
    # json_data = json.loads(impact_graph)
    # for i in treatment_names:
        # convert_to_image(json_data[i])
    st.markdown(
        f"""
    <div style="font-size:16px; font-weight:bold; color:#007BFF;">
        {ate}
    </div>
    """,
        unsafe_allow_html=True
    )
    return ate, graph


def causal_inference(model):
    st.write("Generaitng Causal Inference")
    estimands = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimands,
                                     method_name="backdoor.linear_regression",
                                     effect_modifiers=[],
                                     confidence_intervals=True,
                                     test_significance=True)
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Run interpret() which will print to the redirected output
    estimate.interpret()

    # Reset standard output to its original state
    sys.stdout = old_stdout

    # Capture the interpretation output
    res = new_stdout.getvalue()
    st.markdown(
        f"""
    <div style="font-size:16px; font-weight:bold; color:#007BFF;">
        {manupulate_output(res, st.session_state["treatment_variable"], st.session_state["outcome_variable"], "$")}
    </div>
    """,
        unsafe_allow_html=True
    )


def TraditionalAnalysisApp(df):
    st.header("Dataset Analysis")
    st.write(
        "This section demonstrates the traditional analysis of the data using statistical methods.")

    # Preview the dataset
    st.write("**Preview of the Dataset:**")
    st.write(df.head())

    # Identify feature types (cache to avoid recomputation)
    @st.cache_data
    def get_feature_types(df):
        categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        numerical_features = df.select_dtypes(
            include=['number']).columns.tolist()
        datetime_features = df.select_dtypes(
            include=['datetime64']).columns.tolist()
        return categorical_features, numerical_features, datetime_features

    categorical_features, numerical_features, datetime_features = get_feature_types(
        df)

    st.write("**Feature Types Identified:**")
    st.write(f"- Categorical Features: {categorical_features}")
    st.write(f"- Numerical Features: {numerical_features}")
    st.write(f"- Datetime Features: {datetime_features}")

    # Display summary statistics for numerical and categorical separately
    st.subheader("Summary Statistics:")

    @st.cache_data
    def get_summary_statistics(df, numerical_features, categorical_features):
        numerical_summary = df[numerical_features].describe()
        categorical_summary = df[categorical_features].describe()
        return numerical_summary, categorical_summary

    numerical_summary, categorical_summary = get_summary_statistics(
        df, numerical_features, categorical_features)
    st.write("**Numerical Features:**")
    st.write(numerical_summary)
    st.write("**Categorical Features:**")
    st.write(categorical_summary)

    # Display correlation matrix for numerical features
    if numerical_features:
        st.subheader("Correlation Matrix (Numerical Features):")

        @st.cache_data
        def get_correlation_matrix(df, numerical_features):
            return df[numerical_features].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(get_correlation_matrix(df, numerical_features),
                    annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    if numerical_features:
        st.subheader("Plots:")

        x_variable = st.selectbox(
            "X Variable (Numerical):", numerical_features, key="scatter_x")
        y_variable = st.selectbox(
            "Y Variable (Numerical):", numerical_features, key="scatter_y")

        if x_variable and y_variable:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_variable, y=y_variable, ax=ax)
            ax.set_title(f"Scatter Plot between {x_variable} and {y_variable}")
            st.pyplot(fig)
            correlation_value = df[x_variable].corr(df[y_variable])
            st.write(
                f"Correlation between {x_variable} and {y_variable}: {correlation_value:.2f}")

    # Histogram for numerical features
    if numerical_features:
        variable = st.selectbox(
            "Select variable for histogram (Numerical):", numerical_features, key="histogram_var")

        if variable:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[variable], kde=True, color="blue", ax=ax)
            ax.set_title(f"Histogram of {variable}")
            st.pyplot(fig)

    # Box Plot for categorical and numerical features
    if numerical_features and categorical_features:
        cat_var = st.selectbox(
            "Select categorical variable for box plot:", categorical_features, key="boxplot_cat")
        num_var = st.selectbox(
            "Select numerical variable for box plot:", numerical_features, key="boxplot_num")

        if cat_var and num_var:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=cat_var, y=num_var, ax=ax, palette="Set2")
            ax.set_title(f"Box Plot of {num_var} by {cat_var}")
            st.pyplot(fig)

    # Pair Plot for numerical features
    if len(numerical_features) > 1:
        selected_columns = st.multiselect(
            "Select numerical columns for pair plot (at least 2):",
            numerical_features,
            default=numerical_features[:2],
            key="pairplot_cols"
        )

        if len(selected_columns) > 1:
            st.write("Generating Pair Plot...")
            pairplot_fig = sns.pairplot(
                df[selected_columns], corner=True, diag_kind='kde', plot_kws={'alpha': 0.5})
            st.pyplot(pairplot_fig)

    # Time Series Plot
    if datetime_features:
        time_col = st.selectbox("Select datetime column:",
                                datetime_features, key="time_col")
        num_col = st.selectbox(
            "Select numerical column for time series plot:", numerical_features, key="time_num_col")

        if time_col and num_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=df[time_col], y=df[num_col], ax=ax, color='green')
            ax.set_title(f"Time Series Plot of {num_col} over {time_col}")
            st.pyplot(fig)


def main():
    # st.title("Causal Discovery in Retail")

    st.sidebar.title("Please fill out this information carefully")

    display_analysis = st.sidebar.radio(
        "Do you want to display the dataset analysis?", ("Yes", "No"))
    preprocessed = st.sidebar.radio(
        "Is your dataset preprocessed?", ("Yes", "No"))
    outcome_variable = st.sidebar.text_input(
        "What is your outcome variable?", key="outcome_variable")
    treatment_variable = st.sidebar.text_input(
        "What is your treatment variable?", key="treatment_variable")
    which_model = st.sidebar.radio(
        "Which model would you like to use?", ("causica (complex relations)", "dowhy (simple relations)"), key="model")

    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    st.sidebar.text("Make sure it is in .csv format")

    if uploaded_file:
        st.sidebar.write(f"File '{uploaded_file.name}' uploaded successfully.")

    if st.sidebar.button("Start Prediction") or st.session_state.get('prediction_started', True):
        st.session_state['prediction_started'] = True
        if uploaded_file is None:
            st.sidebar.error("Please upload a dataset to proceed.")
        elif not outcome_variable:
            st.sidebar.error("Please specify an outcome variable.")
        elif not treatment_variable:
            st.sidebar.error("Please specify an treatment_variable.")
        else:
            st.sidebar.success("Starting prediction process...")
            df = pd.read_csv(uploaded_file)
            # Remove ID columns
            df = remove_id_columns(df)

            if preprocessed == "No":
                preprocess(df)
            else:
                preprocess_dates(df)

        # Run the analysis app
            if display_analysis == "Yes":
                TraditionalAnalysisApp(df)

        # Causal Graph
            st.header("Causal Graph")
            st.write("This section demonstrates the causal graph of the data.")

            categorial = df.select_dtypes(
                include=['object', 'category', 'datetime']).columns.tolist()
            df = df.drop(columns=categorial)

            # Making of the di-graph
            # st.text_input("Enter the outcome variable:", key="outcome_variable")
            digraph = get_digraph(df, st.session_state["outcome_variable"])

            # Display the causal graph
            # st.selectbox("Enter the treatment variable:", df.columns, key="treatment_variable")
            if(which_model=="dowhy (simple relations)"):
                causal_model = causal_graph(
                digraph, df, st.session_state["outcome_variable"], st.session_state["treatment_variable"])

            # Causal Inference
            st.subheader("Causal Inference")
            st.write(
                "This section demonstrates the causal inference analysis of the data.")
            if st.session_state["treatment_variable"] and st.session_state["outcome_variable"] and which_model=="dowhy (simple relations)":
                causal_inference(causal_model)
            else:
                # st.table(df.head())
                causica_inference(df, st.session_state["outcome_variable"], st.session_state["treatment_variable"], 1, df.columns)

if __name__ == "__main__":
    main()