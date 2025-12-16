from jax import random, numpy as jnp

from scipy.stats import kendalltau

from visturing.properties.jax.math_utils import kendall_correlation

def test_kendall_correlation():
    """Test the Kendall's tau correlation coefficient."""
    x = random.normal(random.PRNGKey(0), (1000,))
    y = random.normal(random.PRNGKey(2), (1000,))

    assert kendall_correlation(x, y) == kendalltau(x, y)[0]


