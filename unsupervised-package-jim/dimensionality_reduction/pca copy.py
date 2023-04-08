import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA)
    
    Parameters:
    -----------
    n_components : int
        Number of principal components to compute.
        
    Attributes:
    -----------
    components_ : array-like, shape (n_components, n_features)
        Principal components.
        
    explained_variance_ : array-like, shape (n_components,)
        Amount of variance explained by each of the selected components.
        
    explained_variance_ratio_ : array-like, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        
    mean_ : array-like, shape (n_features,)
        Mean of each feature in the original data.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, X):
        """
        Fit the PCA model to the given data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        # Compute the mean of each feature
        self.mean_ = np.mean(X, axis=0)
        
        # Center the data by subtracting the mean
        X_centered = X - self.mean_
        
        # Compute the covariance matrix
        cov = np.cov(X_centered.T)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov)
        
        # Sort the eigenvectors and eigenvalues in descending order of eigenvalue
        eigvecs = eigvecs.T
        idxs = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[idxs]
        eigvals = eigvals[idxs]
        
        # Normalize the eigenvectors
        norm = np.sqrt(np.sum(np.power(eigvecs, 2), axis=1))
        eigvecs /= norm.reshape(-1, 1)
        
        # Store the first n_components eigenvectors
        if self.n_components is not None:
            self.components_ = eigvecs[:self.n_components]
            self.explained_variance_ = eigvals[:self.n_components]
        else:
            self.components_ = eigvecs
            self.explained_variance_ = eigvals
        
        # Compute the percentage of variance explained by each component
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
    def transform(self, X):
        """
        Transform the given data into the principal component space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        # Center the data by subtracting the mean
        X_centered = X - self.mean_
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
