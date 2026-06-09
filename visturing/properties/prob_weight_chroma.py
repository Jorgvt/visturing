import numpy as np
from scipy.special import gamma

sigma_atd = {
    "achrom": 26.8524,
    "red-green": 1.5699,
    "yellow-blue": 2.8205,
}

Hfs = {"achrom": 1.3185*1e4,
    "red-green": 0.0181*1e4,
    "yellow-blue": 0.0480*1e4, }
g_opt = 0.8997
b_opt_T = 0.5197
b_opt_D = 0.4960

def func_weibull_Y(sigma, p, x):

    # 
    # 26.8524    1.5699    2.8205

    x = np.clip(x, 1e-8, None)
    l = sigma/(gamma(1+2/p)-gamma(1+1/p)**2)**0.5
    prob = ((p/l)*(x/l)**(p-1))*np.exp(-((x/l)**p))

    return prob

def gral_gauss(sigma, p, x):

    # 
    # 26.8524    1.5699    2.8205

    a = np.sqrt(gamma(1/p)/gamma(3/p))*sigma
    prob = p/(2*a*gamma(1/p))*np.exp(-(abs(x)/a)**p)

    return prob

def get_weights(YY,
                TT,
                DD,
                Bpp,
                ):

    Hfs = {
        "achrom": 1.3185*1e4,
        "red-green": 0.0181*1e4,
        "yellow-blue": 0.0480*1e4,
    }
    sf = np.array([a for a in sigma_atd.values()])
    Hfs = np.array([a for a in Hfs.values()])
    bf = (Bpp - 1/(2*3)*( sum(np.log2((sf**2)*Hfs)) )) + 0.5*np.log2((sf**2)*Hfs)

    Nf = 2**bf

    dY = YY[-1] - YY[-2]
    PY = func_weibull_Y(sigma_atd["achrom"], g_opt,YY)
    kk = sum(PY*dY)
    pY = (1/kk)*PY
    pY13_norm = (pY**(1/3))/sum( pY**(1/3)*dY )

    dT = TT[-1] - TT[-2]
    PT = gral_gauss(sigma_atd["red-green"], b_opt_T,TT)
    kk = sum(PT*dT)
    pT = (1/kk)*PT
    pT13_norm = (pT**(1/3))/sum( pT**(1/3)*dT )

    dD = DD[-1] - DD[-2]
    PD = gral_gauss(sigma_atd["yellow-blue"], b_opt_D,DD)
    kk = sum(PD*dD)
    pD = (1/kk)*PD
    pD13_norm = (pD**(1/3))/sum( pD**(1/3)*dD )

    Nfc = [Nf[0]*pY13_norm, Nf[1]*pT13_norm, Nf[2]*pD13_norm]
    Nfc = {
        "achrom": Nf[0]*pY13_norm,
        "red-green": Nf[1]*pT13_norm,
        "yellow-blue": Nf[2]*pD13_norm,
    }
    Bfc = {k:np.log2(v) for k, v in Nfc.items()}

    return bf,Nf,Nfc,Bfc
