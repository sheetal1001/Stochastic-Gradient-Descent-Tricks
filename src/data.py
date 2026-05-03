"""
Handles data generation and loading for large-scale, sparse machine learning.
"""
import numpy as np
import scipy.sparse as sp

def generate_sparse_data(n_samples=50000, n_features=5000, density=0.001):
    """
    Generates a high-dimensional sparse dataset to replicate the environment.
        
    Returns:
        X (scipy.sparse.csr_matrix): Sparse feature matrix.
        y (np.array): Labels strictly in {-1, 1}.
    """
    print(f"Generating sparse dataset: {n_samples} samples, {n_features} features...")
    X = sp.random(n_samples, n_features, density=density, format='csr')
    
    # Generate labels using a random true weight vector and some noise
    true_w = np.random.randn(n_features)
    y = np.sign(X.dot(true_w) + np.random.randn(n_samples) * 0.1)
    
    # Ensure no zero labels (must be -1 or 1 for log-loss)
    y[y == 0] = 1 
    return X, y