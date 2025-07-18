"""
Data Processing Utilities for Gram-Schmidt Walk

This module provides functionality to prepare real-world datasets
for use with the Gram-Schmidt Walk algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class DataProcessor:
    """
    Processes real-world datasets for use with Gram-Schmidt Walk algorithm.
    
    This class handles:
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    - Creating appropriate balance vectors
    """
    
    def __init__(self, 
                 handle_missing=True, 
                 handle_categorical=True,
                 handle_outliers=True,
                 scaling='robust',
                 balance_features=True,
                 balance_labels=True,
                 balance_statistics=True,
                 max_dimensions=50):
        """
        Initialize the data processor with specified options.
        
        Args:
            handle_missing: Whether to impute missing values
            handle_categorical: Whether to encode categorical features
            handle_outliers: Whether to use robust scaling for outliers
            scaling: Scaling method ('standard', 'robust', or None)
            balance_features: Whether to include features in balance vectors
            balance_labels: Whether to include labels in balance vectors
            balance_statistics: Whether to include statistical properties 
            max_dimensions: Maximum dimensions for balance vectors
        """
        self.handle_missing = handle_missing
        self.handle_categorical = handle_categorical
        self.handle_outliers = handle_outliers
        self.scaling = scaling
        self.balance_features = balance_features
        self.balance_labels = balance_labels
        self.balance_statistics = balance_statistics
        self.max_dimensions = max_dimensions
        
        self.num_imputer = None
        self.cat_imputer = None
        self.num_scaler = None
        self.cat_encoder = None
        self.pca = None
        self.fitted = False
        
    def fit(self, X, y=None):
        """
        Fit necessary transformers to the data.
        
        Args:
            X: DataFrame or array of features
            y: Optional target variable
            
        Returns:
            self: The fitted processor
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        if hasattr(X, 'select_dtypes'):
            self.num_cols = list(X.select_dtypes(include=['number']).columns)
            self.cat_cols = list(X.select_dtypes(exclude=['number']).columns)
        else:
            self.num_cols = list(range(X.shape[1]))
            self.cat_cols = []
            
        if self.num_cols:
            if self.handle_missing:
                self.num_imputer = SimpleImputer(strategy='median')
                self.num_imputer.fit(X[self.num_cols])
                
            if self.scaling == 'standard':
                self.num_scaler = StandardScaler()
                self.num_scaler.fit(X[self.num_cols])
            elif self.scaling == 'robust' and self.handle_outliers:
                self.num_scaler = RobustScaler()
                self.num_scaler.fit(X[self.num_cols])
                
            if len(self.num_cols) > self.max_dimensions and self.balance_features:
                self.pca = PCA(n_components=min(self.max_dimensions, len(self.num_cols)))
    
                X_num = X[self.num_cols].copy()
                if self.num_imputer:
                    X_num = self.num_imputer.transform(X_num)
                if self.num_scaler:
                    X_num = self.num_scaler.transform(X_num)
                self.pca.fit(X_num)
                
        if self.cat_cols and self.handle_categorical:
            if self.handle_missing:
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
                self.cat_imputer.fit(X[self.cat_cols])
                
            self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat = X[self.cat_cols]
            if self.cat_imputer:
                X_cat = self.cat_imputer.transform(X_cat)
            self.cat_encoder.fit(X_cat)
            
        if y is not None and self.balance_labels:
            y = np.asarray(y)
            if y.ndim == 1:
                self.label_encoder = OneHotEncoder(sparse_output=False)
                self.label_encoder.fit(y.reshape(-1, 1))
            elif y.ndim == 2 and y.shape[1] == 1:
                self.label_encoder = OneHotEncoder(sparse_output=False)
                self.label_encoder.fit(y)
                
        self.fitted = True
        return self
        
    def transform(self, X):
        """
        Transform features using fitted transformers.
        
        Args:
            X: DataFrame or array of features
            
        Returns:
            X_processed: Processed features as numpy array
        """
        if not self.fitted:
            raise ValueError("DataProcessor must be fitted before transform")
            
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.num_cols + self.cat_cols)
            
        if self.num_cols:
            X_num = X[self.num_cols].copy()
            
            if self.num_imputer:
                X_num = self.num_imputer.transform(X_num)
                
            if self.num_scaler:
                X_num = self.num_scaler.transform(X_num)
                
            if self.pca:
                X_num = self.pca.transform(X_num)
        else:
            X_num = np.array([]).reshape(X.shape[0], 0)
            
        if self.cat_cols and self.handle_categorical:
            X_cat = X[self.cat_cols].copy()
            
            if self.cat_imputer:
                X_cat = self.cat_imputer.transform(X_cat)
                
            if self.cat_encoder:
                X_cat = self.cat_encoder.transform(X_cat)
        else:
            X_cat = np.array([]).reshape(X.shape[0], 0)
            
        X_processed = np.hstack([X_num, X_cat])
        
        return X_processed
    
    def fit_transform(self, X, y=None):
        """
        Fit transformers and transform the data.
        
        Args:
            X: DataFrame or array of features
            y: Optional target variable
            
        Returns:
            X_processed: Processed features as numpy array
        """
        return self.fit(X, y).transform(X)
    
    def create_balance_vectors(self, X, y=None):
        """
        Create balance vectors for the Gram-Schmidt Walk algorithm.
        
        Args:
            X: DataFrame or array of raw features
            y: Optional target variable
            
        Returns:
            balance_vectors: Array of normalized balance vectors
        """
        X_processed = self.fit_transform(X, y)
        
        vector_components = []
        
        if self.balance_features:
            vector_components.append(X_processed)
        
        if y is not None and self.balance_labels:
            y = np.asarray(y)
            
            if y.ndim == 1:
                y_encoded = self.label_encoder.transform(y.reshape(-1, 1))
                vector_components.append(y_encoded)
            elif y.ndim == 2 and y.shape[1] == 1:
                y_encoded = self.label_encoder.transform(y)
                vector_components.append(y_encoded)
            else:
                vector_components.append(y)
                
        if self.balance_statistics:
            stat_vectors = []
            
            mean_vector = np.mean(X_processed, axis=0)
            distances = X_processed - mean_vector
            distance_scores = np.linalg.norm(distances, axis=1).reshape(-1, 1)
            stat_vectors.append(distance_scores)
            
            if X_processed.shape[0] < 10000:
                try:
                    k = min(20, X_processed.shape[0] - 1)
                    nn = NearestNeighbors(n_neighbors=k+1)
                    nn.fit(X_processed)
                    distances, _ = nn.kneighbors(X_processed)
                    density_vector = np.mean(distances[:, 1:], axis=1).reshape(-1, 1)
                    stat_vectors.append(density_vector)
                except Exception:
                    pass
            
            if stat_vectors:
                stat_combined = np.hstack(stat_vectors)
                
                col_norms = np.linalg.norm(stat_combined, axis=0)
                col_norms[col_norms < 1e-10] = 1.0
                stat_combined = stat_combined / col_norms
                
                vector_components.append(stat_combined)
        
        if not vector_components:
            raise ValueError("No balance vectors could be created. Check your settings.")
            
        all_vectors = np.hstack(vector_components)
        
        row_norms = np.linalg.norm(all_vectors, axis=1)
        row_norms[row_norms < 1e-10] = 1.0
        
        mask = row_norms > 1.0
        normalized_vectors = all_vectors.copy()
        normalized_vectors[mask] = all_vectors[mask] / row_norms[mask, np.newaxis]
        
        return normalized_vectors


def prepare_dataset(X, y=None, **kwargs):
    """
    Quick helper function to prepare a dataset for Gram-Schmidt Walk.
    
    Args:
        X: Input features (DataFrame or array)
        y: Optional target values
        **kwargs: Additional arguments for DataProcessor
        
    Returns:
        balance_vectors: Normalized balance vectors
    """
    processor = DataProcessor(**kwargs)
    return processor.create_balance_vectors(X, y)


"""
testing
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    processor = DataProcessor()
    balance_vectors = processor.create_balance_vectors(X, y)
    
    print(f"Original data shape: {X.shape}")
    print(f"Balance vectors shape: {balance_vectors.shape}")
    print(f"Max vector norm: {np.max(np.linalg.norm(balance_vectors, axis=1)):.4f}")
"""