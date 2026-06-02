from typing import Sequence
from itertools import product

import numpy as np
from perceptualtests.utils import control_lum_contrast, noise, freqspace, spatio_temp_freq_domain
from perceptualtests.color_matrices import Matd2xyz, Mxyz2atd, Mxyz2ng, gamma

import cmath
def noise(fx2,fy2,fm,fM,angle,delta_a):

    # Noise in frequency domain
    #fm = 10
    #fM = 20
    #angle = 90
    a_m = -delta_a/2
    a_M = delta_a/2

    f = np.sqrt(fx2**2+fy2**2)
    a = 180*np.arctan2(fy2,fx2)/np.pi

    #print(np.min(a),np.max(a))

    F_noise_f = 1*((f>fm) & (f<fM))
    F_noise_a = 1*(((a>a_m+angle) & (a<a_M+angle)) | ((a>a_m+angle+180) & (a<a_M+angle+180)) | ((a>a_m+angle-180) & (a<a_M+angle-180)) )   

    F_noise = F_noise_f*F_noise_a
    if (F_noise==0).all():
        freq = np.random.uniform(fm, fM, size=(10,))
        theta = np.random.uniform(a_m, a_M, size=(10,))
        phases = np.random.uniform(0, 2*np.pi, size=(10,))

        x, y, t, _, _, _ = spatio_temp_freq_domain(len(fx2), len(fy2), 1, np.abs(np.min(fx2))*2, np.abs(np.min(fy2))*2, 1)
        nx = np.zeros_like(x)
        for f, t, p in product(freq, theta, phases):
            fx = f*np.cos(t)
            fy = f*np.sin(t)
            nx += np.sin(fx*x + fy*y + p)
        nf = np.fft.fft2(nx)
        nf = np.fft.fftshift(nf)
        F_noise = 1*(np.abs(nf) != 0)

    else:
        # G = np.exp(-((fx-fx0)**2/sigmas_f[k]**2 +(fy-fy0)**2/sigmas_f[k]**2))
        nf = F_noise*np.exp(cmath.sqrt(-1)*(2*np.pi*np.random.rand(fx2.shape[0], fx2.shape[1])))
        nx = np.fft.ifft2(np.fft.ifftshift(nf)).real

    return nx, nf, F_noise

def gaussian(x, y, sigma, A=1):
    return A*np.exp(-((x-x.mean())**2+(y-y.mean())**2)/(2*sigma**2))

def create_mask(x, y, x0, y0, R0, sig):
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    RC = 0.5*(1+np.cos(np.pi*(r-R0)/sig))
    RC = np.where(r > (sig+R0), 0, RC)
    RC = np.where(r<R0, 1, RC)
    return RC

def prepare_color(img,
                  c: int, # 1 achrom 2 red-green 3 yellow-blue
                  L: float, # Luminance
                  ):
    a = img
    a = a[...,None].repeat(3, axis=-1)
    if c == 1:
        a[...,1:] = 0
    elif c == 2:
        a[...,2] = 0
        a[...,1] -= L
    elif c == 3:
        a[...,1] = 0
        a[...,2] -= L
    a = a @ Matd2xyz.T @ Mxyz2ng.T
    a = np.clip(a, a_min=0., a_max=np.inf)
    a = np.power(a, gamma)
    return a

def generate_noise(img_size: Sequence[int],
                   fs: int,
                   fm: float | None = None,
                   fM: float | None = None,
                   theta: float = 0,
                   delta_theta: float = 0,
                   sigma_mask: float | None = None,
                   R0: float = 0,
                   N: int | None = None, # number of stimuli to generate
                   L: float = 0.5, # Luminance
                   C: float = 0.1, # Contrast
                   c: int = 1, # 1 achrom 2 red-green 3 yellow-blue
                   freqs: Sequence[float] | None = None, # Array of frequencies to use
                   ):
    """Returns both the noises and their corresponding frequencies."""

    x, y, t, ffx, ffy, fft = spatio_temp_freq_domain(*img_size, 1, fs, fs, 1)
    if freqs is not None:
        fs_test = np.array(freqs)
        N = len(fs_test)
        if fM is None:
            fM = np.max(fs_test)
    else:
        if fm is None or fM is None or N is None:
            raise ValueError("Either 'freqs' must be provided, or all of 'fm', 'fM', and 'N' must be provided.")
        fs_test = np.linspace(fm, fM, num=N+1)[:-1]
    
    delta_f_sampling = 3*fs_test/fM
    
    if sigma_mask is not None:
        # mask = gaussian(x, y, sigma=sigma_mask)
        mask = create_mask(x, y, x0=0.5, y0=0.5, sig=sigma_mask, R0=R0)
        # return mask
    else: mask = 1

    noises = np.empty((N, *img_size))
    for i, (f, delta) in enumerate(zip(fs_test, delta_f_sampling)):
        if f == 0:
            noises[i] = np.zeros_like(noises[i])
        else:
            fm_i = f - delta
            fM_i = f + delta
            nx, nf, F_noise = noise(ffx, ffy, fm=fm_i, fM=fM_i, angle=theta, delta_a=delta_theta)

            nx = nx*mask
            noises[i] = nx
    
    noises = np.array([control_lum_contrast(noise, L=L, C=C) for noise in noises])
    noises = np.array([prepare_color(noise, c=c, L=L) for noise in noises])
    return noises, fs_test


def generate_plain(img_size: Sequence[int],
                   L: float, # Luminance
                   ):
    return generate_noise(img_size, fs=1, L=L, C=0, c=1, N=1, fm=3, fM=4)[0]

def generate_noise_iters(img_size: Sequence[int],
                   fs: int,
                   fm: float | None = None,
                   fM: float | None = None,
                   theta: float = 0,
                   delta_theta: float = 0,
                   sigma_mask: float | None = None,
                   R0: float = 0,
                   N: int | None = None, # number of stimuli to generate
                   L: float = 0.5, # Luminance
                   C: float = 0.1, # Contrast
                   c: int = 1, # 1 achrom 2 red-green 3 yellow-blue
                   n_iters: int = 1, # Number of iterations
                   freqs: Sequence[float] | None = None, # Array of frequencies to use
                   ):
    """Returns both the noises and their corresponding frequencies."""

    if freqs is not None:
        N = len(freqs)
    elif N is None:
        raise ValueError("Either 'freqs' or 'N' must be provided.")
        
    stimuli = np.empty(shape=(n_iters, N, *img_size, 3))
    for i in range(n_iters):
        s, freqs_out = generate_noise(img_size, fs=fs, fm=fm, fM=fM, sigma_mask=sigma_mask, R0=R0, N=N, theta=theta, delta_theta=delta_theta, L=L, C=C, c=c, freqs=freqs)
        stimuli[i] = s
    return stimuli, freqs_out

