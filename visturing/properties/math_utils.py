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
