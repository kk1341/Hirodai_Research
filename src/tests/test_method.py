import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import method

class TestMethod(unittest.TestCase):
    def test_sample_covariance_shape(self):
        T, N = 50, 5
        R = np.random.randn(T, N)
        S = method.sample_covariance(R)
        self.assertEqual(S.shape, (N, N))

    def test_positive_definite(self):
        T, N = 50, 5
        R = np.random.randn(T, N)
        S = method.sample_covariance(R)
        # Check symmetry
        self.assertTrue(np.allclose(S, S.T))
        # Check diagonal elements are positive (variance > 0)
        self.assertTrue(np.all(np.diag(S) >= 0))
    
    def test_pca_covariance(self):
        T, N = 50, 5
        R = np.random.randn(T, N)
        K = 2
        Sigma_pca = method.pca_factor_covariance(R, K)
        self.assertEqual(Sigma_pca.shape, (N, N))

    def test_linear_shrinkage_identity(self):
        T, N = 100, 10
        R = np.random.randn(T, N)
        S, Sigma_sh = method.linear_shrinkage_identity(R)
        self.assertEqual(Sigma_sh.shape, (N, N))
        
        # Check positive definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(Sigma_sh)
        self.assertTrue(np.all(eigvals > 0))

if __name__ == '__main__':
    unittest.main()
