import jax.numpy as jnp

def pearson_correlation(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    vec1_mean = vec1.mean()
    vec2_mean = vec2.mean()
    num = vec1 - vec1_mean
    num *= vec2 - vec2_mean
    num = num.sum()
    denom = ((vec1-vec1_mean)**2).sum()**(1/2)
    denom *= ((vec2 - vec2_mean) ** 2).sum()**(1/2)
    return num / denom

def kendall_correlation(x, y):
    n = x.shape[0]
    
    # 1. Broadcast to compare every element with every other element
    # shape: (N, 1) - (1, N) -> (N, N)
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    
    # 2. Compute signs
    # sgn(x_i - x_j)
    sign_x = jnp.sign(diff_x)
    sign_y = jnp.sign(diff_y)
    
    # 3. Compute Concordance
    # Result is 1 if concordant, -1 if discordant, 0 if tied
    concordance = sign_x * sign_y
    
    # 4. Sum and Normalize
    # We sum the whole matrix (excluding diagonal where result is 0).
    # Since the matrix is symmetric, we divide by n(n-1) rather than n(n-1)/2
    # to account for the double counting.
    tau = jnp.sum(concordance) / (n * (n - 1))
    
    return tau
