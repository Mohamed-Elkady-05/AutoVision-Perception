import numpy as np

class PCA_FromScratch:

    def __init__(self, n_components):

        # Number of principal components we want to keep
        self.n_components = n_components


    def fit(self, X):

        # Step 1: Compute the mean of each feature
        # Used later to center the data
        self.mean = np.mean(X, axis=0)

        # Step 2: Mean Centering
        # Subtract the mean so the data is centered around zero
        X_centered = X - self.mean

        # Step 3: Compute covariance matrix
        # Covariance matrix describes how features vary together
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 4: Eigen decomposition
        # Eigenvectors = directions of maximum variance
        # Eigenvalues = amount of variance in each direction
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 5: Sort eigenvectors by descending eigenvalues
        # Largest variance components come first
        idx = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, idx]

        # Step 6: Select top k eigenvectors
        # These become the principal components
        self.components = eigenvectors[:, :self.n_components]


    def transform(self, X):

        # Step 7: Center the data using training mean
        X_centered = X - self.mean

        # Step 8: Project data onto principal components
        # X_reduced = X_centered * W
        return np.dot(X_centered, self.components)
