from jax import numpy as jnp

from scipy.stats import kendalltau

from visturing.properties.jax.math_utils import kendall_correlation

def test_kendall_correlation():
    """Test the Kendall's tau correlation coefficient."""
    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([5, 4, 3, 2, 1])

    assert kendall_correlation(x, y) == kendalltau(x, y)[0]


