import numpy as np

class SVD:
    """
    This class provides an implementation of the Singular Value Decomposition (SVD) algorithm using NumPy.
    It allows for computing the SVD of a given matrix, as well as the truncated SVD.
    
    Attributes:
        n_components (int): The number of components to keep in the truncated SVD.
    
    Methods:
        fit_transform(X): Computes the SVD of a given matrix X and returns its singular values, left and right singular vectors.
        fit_transform_truncated(X): Computes the truncated SVD of a given matrix X and returns its singular values, left and right singular vectors.
    
    """

    def __init__(self, n_components=None):
        """
        Constructor for the SVD class.
        
        Args:
            n_components (int): The number of components to keep in the truncated SVD. Defaults to None.
        """
        self.n_components = n_components

    def fit_transform(self, X):
        """
        Computes the SVD of a given matrix X and returns its singular values, left and right singular vectors.
        
        Args:
            X (numpy.ndarray): The input matrix to compute the SVD of.
            
        Returns:
            tuple: A tuple of the singular values, left singular vectors, and right singular vectors.
        """
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return S, U, Vt.T

    def fit_transform_truncated(self, X):
        """
        Computes the truncated SVD of a given matrix X and returns its singular values, left and right singular vectors.
        
        Args:
            X (numpy.ndarray): The input matrix to compute the truncated SVD of.
            
        Returns:
            tuple: A tuple of the truncated singular values, truncated left singular vectors, and truncated right singular vectors.
        """
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        if self.n_components is None:
            return S, U, Vt.T
        else:
            return S[:self.n_components], U[:, :self.n_components], Vt[:self.n_components, :]

