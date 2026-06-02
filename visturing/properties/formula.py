import numpy as np

def sigm1d(x, x0, s):
    return 1 / (1 + np.exp((x - x0)/s))

def umbinc3(c, cu, k, m, alf, sig):
    term1 = (cu - k*cu**m) * (1 / (1 + np.exp(np.log10(c / (alf*cu)) / sig)))
    term2 = (k * c**m) * (1 - 1 / (1 + np.exp(np.log10(c / (0.9*cu)) / (sig/2))))
    return term1 + term2

# --- iafyb ---
def iafyb(f, C, facfrec, nolin):
    f = np.atleast_1d(facfrec * f).astype(float)
    C = np.atleast_1d(C).astype(float)
    
    f[f==0] += 1e-5
    C[C==0] += 1e-7
    
    lf = len(f)
    lc = len(C)
    
    iaf = np.zeros((lf, lc))
    p = [0.1611, 1.3354, 0.3077, 0.7746]
    
    if np.isscalar(nolin):
        nolin = [nolin, nolin]
    
    nolini = nolin
    nolin = nolini[0]
    
    if nolini[0] == 0 and nolini[1] == 1:
        nolin = 1
    
    if nolin == 1:
        ace = np.zeros((lf, lc))
        for i in range(lf):
            cu = 1 / (100*719.7*sigm1d(f[i], -31.72, 4.13))
            ace[i, :] = umbinc3(C, cu, *p)
        iaf = 1 / ace
    else:
        iaf_val = 100*719.7*sigm1d(f, -31.72, 4.13)
        iaf = np.outer(iaf_val, np.ones(lc))
    
    csfyb = iaf[:, 0]
    
    if nolini[0] == 0 and nolini[1] == 1:
        iafc = np.sum(iaf, axis=1)
        iaf = np.outer(iafc, np.ones(lc))
    
    return iaf, csfyb


def iafrg(f, C, facfrec, nolin):
    f = np.atleast_1d(facfrec * f).astype(float)
    C = np.atleast_1d(C).astype(float)

    f[f==0] += 1e-5
    C[C==0] += 1e-7

    lf = len(f)
    lc = len(C)

    iaf = np.zeros((lf, lc))
    p = [0.0840, 0.8345, 0.6313, 0.2077]

    if np.isscalar(nolin):
        nolin = [nolin, nolin]

    nolini = nolin
    nolin = nolini[0]

    if nolini[0] == 0 and nolini[1] == 1:
        nolin = 1

    if nolin == 1:
        ace = np.zeros((lf, lc))
        for i in range(lf):
            cu = 1 / (100*2537.9*sigm1d(f[i], -55.94, 6.64))
            ace[i, :] = umbinc3(C, cu, *p)
        iaf = 1 / ace
    else:
        iaf_val = 100*2537.9*sigm1d(f, -55.94, 6.64)
        iaf = np.outer(iaf_val, np.ones(lc))

    csfrg = iaf[:, 0]

    if nolini[0] == 0 and nolini[1] == 1:
        iafc = np.sum(iaf, axis=1)
        iaf = np.outer(iafc, np.ones(lc))

    return iaf, csfrg

from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import os

# --- Cargar datos .mat ---
data_path = os.path.dirname(__file__)
data_path = os.path.join(data_path, "data_csf_spatio_temp_chrom.mat")
if os.path.exists(data_path):
    data = loadmat(data_path)
    ffxx = data['ffxx'].squeeze()
    fftt = data['fftt'].squeeze()
    CSFrg_n = data['CSFrg_n'].T  # Transponer para que coincida con (fftt, ffxx)
    CSFyb_n = data['CSFyb_n'].T  # Igual
else:
    raise FileNotFoundError(f"No se encontró {data_path}")

# Interpoladores para los canales cromáticos
interp_rg = RegularGridInterpolator((fftt, ffxx), 220*CSFrg_n,
                                    bounds_error=False, fill_value=np.min(220*CSFrg_n))
interp_yb = RegularGridInterpolator((fftt, ffxx), 220*CSFyb_n,
                                    bounds_error=False, fill_value=np.min(220*CSFyb_n))


def detect_threshold_spatio_temporal_chromatic(fx, fy, ft, color, kind):
    """
    Devuelve el umbral de detección de contraste para frecuencias espaciales y temporales
    y para los tres canales de color (Achromatic, Red-Green, Yellow-Blue).
    """
    f = np.sqrt(fx**2 + fy**2)
    ft_safe = ft if ft != 0 else 1e-13
    f_safe = max(f, 1e-13)

    # --- Achromatic ---
    if color == 1:
        if ft == 0:
            if kind == 1:
                g, fm, l, s, w, os = 330.74, 7.28, 0.837, 1.809, 1, 6.664
                f_val = np.where(f > 0, f, 1e-4)
                CSFT = g*(np.exp(-f_val/fm) - l*np.exp(-(f_val**2/s**2)))
                OE = 1 - w*(4*(1-np.exp(-f_val/os))*fx**2*fy**2)/(f_val**4)
                delta_C = 1 / (CSFT * OE)
            elif kind in [2, 3]:
                ft_safe = abs(ft) + 0.1*f_safe
                CSFet = 4*np.pi**2 * f_safe * abs(ft_safe) * np.exp(-4*np.pi*(ft_safe + 2*f_safe)/45.9) * \
                        (6.1 + 7.3 * abs(np.log10(abs(ft_safe)/(3*f_safe)))**3)
                delta_C = 1 / CSFet
        else:  # ft != 0
            ft_safe = abs(ft) + 0.1*f_safe
            CSFet = 4*np.pi**2 * f_safe * abs(ft_safe) * np.exp(-4*np.pi*(ft_safe + 2*f_safe)/45.9) * \
                    (6.1 + 7.3 * abs(np.log10(abs(ft_safe)/(3*f_safe)))**3)
            delta_C = 1 / CSFet

    # --- Red-Green ---
    elif color == 2:
        if kind == 4:  # usa iafrg
            iaf_rg0, csf_c0 = iafrg(0, 0.1, 1, [0,0,0])
            iaf_rg, csf_c = iafrg(f_safe, 0.1, 1, [0,0,0])
            csfrg = csf_c
            fact_rg = 0.75
            max_CSF_achro = 201.3
            csfrg = fact_rg * max_CSF_achro * csfrg / np.max(csf_c0)
            delta_C = 1 / csfrg
        else:  # kind==5 -> interpolación de .mat
            delta_C = 1 / interp_rg([[ft_safe, f_safe]])[0]

    # --- Yellow-Blue ---
    elif color == 3:
        if kind == 4:  # usa iafyb
            iaf_yb0, csf_c0 = iafyb(0, 0.1, 1, [0,0,0])
            iaf_yb, csf_c = iafyb(f_safe, 0.1, 1, [0,0,0])
            csfyb = csf_c
            fact_yb = 0.55
            max_CSF_achro = 201.3
            csfyb = fact_yb * max_CSF_achro * csfyb / np.max(csf_c0)
            delta_C = 1 / csfyb
        else:  # kind==5 -> interpolación de .mat
            delta_C = 1 / interp_yb([[ft_safe, f_safe]])[0]

    return delta_C

def incremental_threshold_spatio_temp(fx, fy, ft, C, fxm, fym, ftm, Cm, color, kind):
    """
    Returns the contrast incremental threshold of a test (fx, fy, ft, C)
    on top of a background (fxm, fym, ftm, Cm) for a given color channel and kind of CSF.
    
    Returns:
        Delta_C: final incremental threshold
        Delta_C_abs: detection threshold
        Delta_C_t: threshold contribution due to the test
        Delta_C_m: threshold contribution due to the mask
        g: masking factor
    """
    # 1️⃣ Detection threshold
    Delta_C_abs = detect_threshold_spatio_temporal_chromatic(fx, fy, ft, color, kind)
    
    # 2️⃣ Increment due to the contrast of the test
    f = np.sqrt(fx**2 + fy**2)
    
    if kind > 3:  # Chromatic channels
        f_thresh = 4
        f = max(f, f_thresh)
    else:
        f_peq = 0.03
        f = max(f, f_peq)
    
    # Factor g for contrast nonlinearity
    exponent = (0.8 * f**1.7) / (0.5 + f**1.7)
    g = ((-0.03 * np.log10(f) + 0.3) * C**exponent - Delta_C_abs) / \
        (((-0.03 * np.log10(f) + 0.3) * (1/Delta_C_abs))**(- (0.5 + f**1.7)/(0.8*f**1.7)) + C)
    factor = max(0.6, 2*(f/25))
    g *= factor
    
    Delta_C_t = Delta_C_abs + C*g
    
    # 3️⃣ Elevation due to the background mask
    Delta_C_m = threshold_mask(fx, fy, ft, fxm, fym, ftm, Cm)
    
    # 4️⃣ Final incremental threshold
    Delta_C = Delta_C_t * Delta_C_m
    
    return Delta_C, Delta_C_abs, Delta_C_t, Delta_C_m, g

def threshold_mask(fx, fy, ft, fxm, fym, ftm, Cm):
    """
    Computes the effect of the background mask on the threshold.
    """
    f = np.sqrt(fx**2 + fy**2)
    fm = np.sqrt(fxm**2 + fym**2)
    
    f = f if f != 0 else 1e-6
    fm = fm if fm != 0 else 1e-6
    ft = ft if ft != 0 else 1e-6
    ftm = ftm if ftm != 0 else 1e-6
    
    theta = np.degrees(np.arctan2(fy, fx))
    thetam = np.degrees(np.arctan2(fym, fxm))
    
    A = 0.5
    k = 1
    c_m_0 = 1.8
    sigma_f = 0.25
    sigma_ft = 0.5
    sigma_theta = 45.0
    exp_theta = 1
    
    dif_angle = min(abs(theta - thetam), abs(theta - 180 - thetam), abs(theta + 180 - thetam))
    
    n = A * np.exp(-(np.log10(f) - np.log10(fm))**2 / sigma_f**2) * \
            np.exp(-(np.log10(ft) - np.log10(ftm))**2 / sigma_ft**2) * \
            np.exp(-(dif_angle/sigma_theta)**exp_theta)
    
    contrast = 100.0 * Cm
    Delta_Cm = k * (contrast/c_m_0)**n if contrast >= c_m_0 else 1.0
    
    return Delta_Cm
