# 1
import numpy as np

def run_point1(nrow=2,ncols=2):
    # Define matrix rows and cols
    n_rows = nrow
    n_cols = ncols

    # Creating a random matrix
    random_matrix = np.random.rand(n_rows,n_cols)

    # Get Matrix Rank
    rank = np.linalg.matrix_rank(random_matrix)
    print("Matrix rank: ",rank)

    # Get Matrix Trace
    trace = np.trace(random_matrix)
    print("Matrix Trace", trace)

    # Get Matrix Determinant and Inverse, we can get it only for square matrix
    if n_rows == n_cols:
        determinant = np.linalg.det(random_matrix)
        print("Matrix Determinant", determinant)
        inverse = np.linalg.inv(random_matrix)
        print("Matrix Inverse",inverse)
    else:
        print("Matrix Determinant only can compute for square matrix")

    # Get eigenvalues and eigenvectors of A’A and AA’ related? What interesting differences can you notice between both? ##
    AtransposedxA = random_matrix.T.dot(random_matrix)
    AxAtransposed = random_matrix.dot(random_matrix.T)
        
    eigval_AtransposedxA, eigvec_AtransposedxA = np.linalg.eig(AtransposedxA)
    eigval_AxAtransposed, eigvec_AxAtransposed = np.linalg.eigh(AxAtransposed)
        
    print("\n eigenvectors of A’A \n",eigvec_AtransposedxA)
    print("\n eigenvectors of AA’ \n",eigvec_AxAtransposed)
    print("\n eigenvalues of A’A \n",eigval_AtransposedxA)
    print("\n eigenvalues of AA' \n",eigval_AxAtransposed)   

    print("\nEigenvectos and eigenvalues for both A'A and AA' are related because their come from dot product of the same matrix  \n")
    print("An interesting diferences,  A’A correspond to matrix cols size and AA' correspond to matrix rows size \n")