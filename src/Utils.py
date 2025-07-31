import re


def manupulate_output(res, treatment_variable, outcome_variable, unit):
    pattern = r"[-+]?\d*\.\d+"
    match = re.search(pattern, res)
    if match:
        extracted_value = float(match.group())
        print("Extracted Value:", extracted_value)
    else:
        print("No match found.")
        extracted_value = 0.0  # Default value if no match found
    
    estimated_effect = extracted_value

    # Define the user-specified range for treatment variable (e.g., $1 to $2)
    treatment_start = 1  # e.g., $1
    treatment_end = 2    # e.g., $2

    # Calculate the difference in treatment range (e.g., $2 - $1)
    treatment_range = treatment_end - treatment_start

    # Adjust the effect to match the user-defined range
    adjusted_effect = estimated_effect * treatment_range

    # Format the output to be more interpretable
    return (f"Increasing the treatment variable [{treatment_variable}] from {treatment_start}{unit} to {treatment_end}{unit} "
            f"causes an increase of approximately {adjusted_effect:.2f}$ in the expected value of the outcome [{outcome_variable}], "
            f"over the data distribution/population represented by the dataset.")


def remove_id_columns(dataframe):
    columns_to_remove = []

    for col in dataframe.columns:
        # Check if 'id' is in the column name (case-insensitive) and it's likely an ID column
        if 'id' in col.lower() and (col.lower().endswith('id') or col.lower().startswith('id')):
            columns_to_remove.append(col)
            continue

        # Check if values are mostly unique (e.g., >95% unique) and likely an identifier
        # But exclude columns with very few unique values (likely categorical)
        unique_ratio = dataframe[col].nunique() / len(dataframe)
        if unique_ratio > 0.95 and dataframe[col].nunique() > 10:
            columns_to_remove.append(col)
            continue

        # Check if values are monotonically increasing or decreasing (only for numeric columns)
        # This catches index-like columns
        if dataframe[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            if len(dataframe) > 1 and (dataframe[col].is_monotonic_increasing or dataframe[col].is_monotonic_decreasing):
                # Additional check: if it's a simple sequence like 1,2,3,4... or looks like an index
                if dataframe[col].nunique() == len(dataframe):  # All values are unique
                    columns_to_remove.append(col)

    # Drop identified columns
    return dataframe.drop(columns=columns_to_remove, errors='ignore')