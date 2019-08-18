from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Basic data transformation pipeline with an imputer and a scaler transform
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std scaler', StandardScaler()),
])

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd

X = pd.DataFrame(
    {'city': ['London', 'London', 'Paris', 'Sallisaw'],
     'title': ["His Last Bow", "How Watson Learned the Trick", "A Moveable Feast", "The Grapes of Wrath"],
     'expert_rating': [5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3]})

column_trans = ColumnTransformer([
    ('city category', OneHotEncoder(dtype='int'), ['city']),        # One Hot Encoder requires 2D data as an input, thus we have to pass the column name as a list of strings, 
                                                                    # as is the case with most transformers.
    ('title bow', CountVectorizer(), 'title')],                     # CountVectorizer takes a 1D array as input, thus the column is passed as a string 
    remainder='drop') # The 'remainder' parameter determines whether to ignore(drop) the remaining columns. The columns can be kept by using remainder='passthrough'
    # The remainder can also be set to an estimator to transform the remaining columns
    # remainder=MinMaxScaler())
column_trans.fit(X)

print(column_trans.get_feature_names())

# The make_column_transformer function is a useful alternative as it automatically assigns names
col_tran = make_column_transformer(
    (OneHotEncoder(),['city']),
    (CountVectorizer(), 'title'),
    remainder=MinMaxScaler())
print(col_tran)
    