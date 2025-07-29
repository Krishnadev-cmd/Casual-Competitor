import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import numpy as np

def clean(df,drop, stratergy):
    #find columns with missing values
    nan_catergorical_values=[(col,v) for col,v in df.isna().sum().items() if v >0 and df[col].dtype == 'object']
    nan_numerical_values=[(col,v) for col,v in df.isna().sum().items() if v >0 and df[col].dtype != 'object']

    if drop:
        df=df.dropna()

    if not drop and nan_numerical_values and stratergy in ['mean', 'median', 'most_frequent']:
        imp=SimpleImputer(strategy=stratergy,missing_values=np.nan)
        nan_numerical_col=[col[0] for col in nan_numerical_values]
        imputed_data_num=imp.fit_transform(df[nan_numerical_col])
        df[nan_numerical_col]=imputed_data_num

    if not drop and nan_catergorical_values:
        imp=SimpleImputer(strategy='most_frequent',missing_values=np.nan)
        nan_categorical_col=[col[0] for col in nan_catergorical_values]
        imputed_data_cat=imp.fit_transform(df[nan_categorical_col])
        df[nan_categorical_col]=imputed_data_cat
    return df
#Encoding function
def encoding(df,code):
    if code=='Label':
        le=LabelEncoder()
        cols=df.select_dtypes(include=['object']).columns
        for col in cols:
            df[col]=le.fit_transform(df[col])

    elif code=='OneHot':
        hot=OneHotEncoder(sparse_output=False)
        cols=df.select_dtypes(include=['object']).columns
        for col in cols:
            df[col]= hot.fit_transform(df[[col]])

    #convert Boolean columns to integers
    bool=df.select_dtypes(include=['bool']).columns
    for col in bool:
        df[col]=df[col].map({True:1,False:0})
    return df

#Scaling function
def scaler(df,scale):
    numerical_cols=df.select_dtypes(include=[np.number]).columns
    if scale=='Standard':
        scaler=StandardScaler()
        df[numerical_cols]=scaler.fit_transform(df[numerical_cols])
    elif scale=='MinMax':
        scaler=MinMaxScaler()
        df[numerical_cols]=scaler.fit_transform(df[numerical_cols])
    return df

def handle_outliers(df):
    """
    Function to handle outliers - IQR method
    """
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

def preprocess(df, drop=False, stratergy='mean', code='Label', scale='Standard'):
    df = clean(df, drop, stratergy)
    df = encoding(df, code)
    df = scaler(df, scale)
    df = handle_outliers(df)
    return df

def determin_types(df, column):
    unique_values=df[column].nunique()
    if pd.api.types.is_numeric_dtype(df[column]):
        if unique_values==2:
            return 'Boolean'
        elif unique_values < 20:
            return 'Categorical'
        else:
            return 'Continuous'
    else:
        return 'Categorical'
def determine_number_of_epochs(df):
    print(300/((len(df)/2048)*0.2))
    return int(300/((len(df)/2048)*0.2))