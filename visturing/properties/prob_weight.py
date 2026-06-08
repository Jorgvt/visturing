import numpy as np
from scipy.special import gamma

params_logsigma = {
    "achrom": {"A": -0.59, "alpha": 1},
    "red-green": {"A": -1.38, "alpha": 1},
    "yellow-blue": {"A": -1.21, "alpha": 1},
}

params_weibull = {
    "achrom": {"k": 1.18},
    "red-green": {"k": 0.83},
    "yellow-blue": {"k": 0.87},
}

PARAMS_PROB = {
    "logsigma": params_logsigma,
    "weibull": params_weibull,
}

def log_sigma(f, A, alpha):
    return A - alpha*np.log10(f)

def p_C(C, f, k, A, alpha):
    sigma_f = 10**(log_sigma(f, A, alpha))
    l = sigma_f/((gamma(1+2/k)) - (gamma(1+1/k)**2))**(1/2)
    return ((k/l)*(C/l)**(k-1))*np.exp(-(C/l)**k)

def Hf(f, Cs, k, A, alpha):
    Cs = Cs.copy()
    dCf = Cs[1] - Cs[0]
    if Cs[0] == 0:
        Cs[0] = Cs[1]
    if len(f.shape) == 0:
        return np.sum(p_C(Cs, f, k, A, alpha)**(1/3) * dCf)**3
    else:
        Hfs = []
        for i, f_ in enumerate(f):
            blah = np.sum(p_C(Cs, f_, k, A, alpha)**(1/3) * dCf)**3
            Hfs.append(blah)

        return np.array(Hfs)

def b_f_opt(f, Bpp, Cs, k, A, alpha):
    """Produces the value for only one channel, doesn't normalize them together."""

    const =  Bpp - (1/(2))*np.mean(np.log2(10**(log_sigma(f, A, alpha))**2*Hf(f, Cs, k, A, alpha)))
    return const + (1/2)*np.log2((10**log_sigma(f, A, alpha))**2*Hf(f, Cs, k, A, alpha))

# b_f_opts = {}
# for c in ["A", "T", "D"]:
#     b_f_opts_ = []
#     for f in fs:
#         b_f_opts_.append(b_f_opt(f, Bpp, Cs, **params_weibull[c], **params_logsigma[c]))
#     b_f_opts_ = np.array(b_f_opts_)
#     b_f_opts[c] = b_f_opts_

def b_f_opt_colors(fs, Bpp, Cs, pms_weibull, pms_logsigma):
    """Normalizes all channels together."""

    kks = []
    for c in ["achrom", "red-green", "yellow-blue"]:
        p_w = pms_weibull[c]
        p_ls = pms_logsigma[c]

        kk = (10**log_sigma(fs, **p_ls))**2 * Hf(fs, Cs, **p_w, **p_ls)
        kk = np.log2(kk)

        kks.append(kk)
    const = Bpp - (1/2)*np.mean(kks)

    bs = {}
    for c in ["achrom", "red-green", "yellow-blue"]:
        p_w = pms_weibull[c]
        p_ls = pms_logsigma[c]

        bs[c] = const + (1/2) * np.log2( (10**log_sigma(fs, **p_ls))**2 * Hf(fs, Cs, **p_w, **p_ls))

    return bs, const



def get_weights(freqs,
                Cs,
                Bpp: int,
                ):

    strip = False
    if len(Cs) == 1:
        Cs = np.array([0, Cs[0]])
        strip = True
    bs, cte = b_f_opt_colors(freqs, Bpp, Cs, params_weibull, params_logsigma)

    N_f_opts = {k:2**v for k,v in bs.items()}

    N_norm = np.concatenate(list(N_f_opts.values())).sum()
    dCf = Cs[1] - Cs[0]
    Cs_ = Cs.copy()
    Cs_[0] = Cs_[1]
    Lambda_opts = {}
    for ic, color in enumerate(["achrom", "red-green", "yellow-blue"]):
        p_w = params_weibull[color]
        p_ls = params_logsigma[color]
        Lambda_opt = np.empty(shape=(len(Cs), len(freqs)))

        for i, c in enumerate(Cs_):
            for j, f in enumerate(freqs):
                Lambda_opt[i,j] = N_f_opts[color][j]/N_norm * (p_C(c, f, **p_w, **p_ls)**(1/3))/(p_C(Cs_, f, **p_w, **p_ls)**(1/3)*dCf).sum()
        if strip: Lambda_opts[color] = Lambda_opt[1:].squeeze()
        else: Lambda_opts[color] = Lambda_opt

    return Lambda_opts


