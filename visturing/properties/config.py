import numpy as np

# Default configurations for each property

default_prop2_config = {
    "img_size": (128, 128),
    "square_size": (64, 64),
}

default_prop3_4_config = {
    "img_size": (128, 128),
    "freqs": np.arange(0, 16, step=1),
    "L": 40.0,
    "C": 0.01,
    "fs": 128,
    "sigma_mask": 0.3,
    "n_iters": 2,
    "theta": 0.0,
    "delta_theta": 30.0,
}

default_prop5_config = {
    "img_size": (128, 128),
    "freqs": np.arange(1, 17, step=1),
    "freqs_mask": np.array([2, 4, 6, 8, 10]),
    "L": 40.0,
    "C": 0.025,
    "C_mask": 0.0125 / 0.3,
    "fs": 128,
    "sigma_mask": 0.3,
    "n_iters": 2,
    "theta": 0.0,
    "delta_theta": 30.0,
}

default_prop6_7_config = {
    "img_size": (128, 128),
    "freqs": np.arange(1, 17, step=1),
    "L": 40.0,
    "Cs": np.linspace(0.01, 0.2, num=10),
    "fs": 128,
    "sigma_mask": 0.3,
    "n_iters": 2,
    "theta": 0.0,
    "delta_theta": 30.0,
}

default_prop8_config = {
    "img_size": (128, 128),
    "freqs": np.arange(1, 17, step=1),
    "freqs_mask": np.array([2, 4, 6, 8, 10]),
    "L": 40.0,
    "Cs": np.linspace(0.01, 0.2, num=10),
    "Cs_mask": np.linspace(0.05, 0.2, num=4) / 0.3,
    "fs": 128,
    "sigma_mask": 0.3,
    "n_iters": 2,
    "theta": 0.0,
    "theta_mask": np.array([0.0]),
    "delta_theta": 30.0,
}

default_prop9_config = {
    "img_size": (128, 128),
    "freqs": np.array([2, 4, 6, 8, 10]),
    "freqs_mask": np.arange(1, 17, step=1),
    "L": 40.0,
    "Cs": np.array([0.025]),
    "Cs_mask": np.array([0.0125]) / 0.3,
    "fs": 128,
    "sigma_mask": 0.3,
    "n_iters": 2,
    "theta": 0.0,
    "theta_mask": np.array([0.0]),
    "delta_theta": 30.0,
}

default_prop10_config = {
    "img_size": (128, 128),
    "freqs": np.arange(2, 17, step=1),
    "freq_mask": 3.0,
    "L": 40.0,
    "Cs": np.linspace(0.01, 0.2, num=11)[1:],
    "Cs_mask": np.linspace(0.05, 0.2, num=4) / 0.3,
    "fs": 128,
    "sigma_mask": 0.3,
    "n_iters": 2,
    "theta": 0.0,
    "thetas_mask": np.linspace(0, 180, num=9)[:-1],
    "delta_theta": 30.0,
}

# A central registry mapping property names to their default configs
DEFAULT_CONFIGS = {
    "prop2": default_prop2_config,
    "prop3_4": default_prop3_4_config,
    "prop5": default_prop5_config,
    "prop6_7": default_prop6_7_config,
    "prop8": default_prop8_config,
    "prop9": default_prop9_config,
    "prop10": default_prop10_config,
}
