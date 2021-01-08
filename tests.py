"""
    This python file contains a script that verifies the implementations of functions wrote in
    ParalleRegression are correct.
"""
from __future__ import print_function
import ParallelRegression as PR
import numpy as np
from pyspark import SparkContext
import unittest



class ParallelRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
            called only once before all the tests
        """
        super(ParallelRegressionTest, cls).setUpClass()
        cls.sc = SparkContext(appName='Parallel Regression Test')

    @classmethod
    def tearDownClass(cls):
        """"
            called only once after all the tests
        """
        super(ParallelRegressionTest, cls).tearDownClass()
        cls.sc.stop()

    def test_localGradient(self):
        """
            verifies that computation of the gradient is correct-or, at least, is consistent
            with implementation of f. This function imports f and localGradient from ParallelRegression.py
            as well as function estimateGrad.

            correctness can be tested by confirming that localGradient agrees with the estimate produced by
            estimateGrad when delta is small.
        """
        x = np.array([np.cos(t) for t in range(5)])
        y = 1.0
        beta = np.array([np.sin(t) for t in range(5)])
        localGradient = PR.localGradient(x, y, beta)
        estimatedGrad = PR.estimateGrad(lambda beta: PR.f(x, y, beta), beta, 0.00001)

        print('\nVerifies that computation of the gradient is correct-or, at least, is consistent \n')
        print(f'local gradient: {localGradient}')
        print(f'estimated gradient: {estimatedGrad}')

        for actualVal, expVal in zip(localGradient, estimatedGrad):
            self.assertLess(abs(actualVal - expVal), 0.0001)

    def testGradient(self):
        """
            verifies whether gradient estimate is correct, 'data/small.test' as a dataset. This function
            imports readData, gradient, estimateGrad and F from ParallelRegression.py.
        """
        lam = 1.0
        beta = np.array([np.sin(t) for t in range(9)])
        data = PR.readData('data/small.test', self.sc)
        actualGrad = PR.gradient(data, beta, lam)
        expGrad = PR.estimateGrad(lambda beta: PR.F(data, beta, lam), beta, 0.0000001)

        print('\n Verifies whether gradient estimate is correct \n')
        print(f'actual gradient: {actualGrad}')
        print(f'expected gradient: {expGrad}')

        for actualVal, expVal in zip(actualGrad, expGrad):
            self.assertLess(abs(actualVal - expVal), 0.0001)


if __name__ == '__main__':
    unittest.main()
