import numpy as np
experiment_parameters = {
    'angles': (np.linspace(-10, 10, 3) / 360) * 2 * np.pi,
    'c_ref': 1540.,
    'depth': 5e-2,
    'sampling_frequency': 20.832e6,
    'carrier_frequency': 5.208e6,
    'fractional_bandwidth': 0.75,
    'probe_name': 'GE9LD',
    'probe_pitch': 230e-6,
    'probe_n_elem': 192,
    'probe_elem_width': 0.9 * 230e-6,
    'tukey_parameter': 0.125
}
