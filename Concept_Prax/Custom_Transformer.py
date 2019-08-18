from sklearn.base import BaseEstimator, TransformerMixin

class Custom_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, custom_specs=False):
        self.custom_specs = custom_specs
    
    def fit():
        return self

    def transform(self, X, y=None):
        """
        Here is where the logic for a custom transformation goes. The 'custom_specs' attribute can be used as a flag to decide if certain
        operations are to be performed on this dataset. For example there could be a specification for standard normalization of all features, 
        but there may be situations where we want to perform all transformation operations EXCEPT that standard normalization. In that case the flag is left to False
        """
        if self.custom_specs == True:
            #custom logic
            pass
        return np.c_(training_set, new_cols)