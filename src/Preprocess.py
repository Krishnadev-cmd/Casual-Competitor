import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

def preprocess(df,drop, stratergy):
    #find columns with missing values
    nan_catergorical_values=[(col,v) for col,v in df.isna().sum().items() if v >0 and df[col].dtype == 'object']
    nan_numerical_values=[(col,v) for col,v in df.isna().sum().items() if v >0 and df[col].dtype != 'object']

    if drop:
        df=df.dropna()

    if not drop and nan_catergorical_values:
        imp=SimpleImputer(strategy=stratergy,missing_values=np.nan)
        nan_numerical_col=[col[0] for col in nan_numerical_values]
        imputed_data=imp.fit_transform(df[nan_numerical_col])
        df=pd.DataFrame(imputed_data,columns=nan_numerical_col)

