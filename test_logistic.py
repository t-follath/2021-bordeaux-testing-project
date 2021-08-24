import logistic
import pytest
import numpy as np

@pytest.mark.logistic
@pytest.mark.parametrize("x,r, expected", [(0.1,2.2, 0.198), (0.2,3.4, 0.544), (0.75, 1.7, 0.31875)])
def test_logistic(x, r, expected):
	assert np.isclose(logistic.logistic_function(x,r), expected)

@pytest.mark.logistic	
@pytest.mark.parametrize("x,r, it, expected", [(0.1,2.2,1, [0.198]), (0.2,3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]), (0.75, 1.7, 2, [0.31875, 0.369152])])
def test_iteration(x, r, it, expected):
	assert np.allclose(logistic.iterate_f(it, x,r), expected, atol=1e-6)
	#plot_trajectory(1, 2.2, 0.1)

@pytest.mark.fuzzing
@pytest.mark.parametrize("x", np.random.uniform(0,1,10))	
@pytest.mark.parametrize("it",  np.arange(10,50,10))	
def test_convergence(x, it, r= 1.5, expected = 1/3.0):
	assert np.isclose(logistic.iterate_f(it, x,r)[-1], expected, atol=1e-6)

