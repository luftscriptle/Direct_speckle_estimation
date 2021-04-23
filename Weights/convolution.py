import numpy as np
import Samuel.core.grid as grid
import scipy.signal as sp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt


def get_convolution_operator(d_grid, theta_rx, theta, params, ratio=0.75, n_sigma=4, debug=False):
    carrier_frequency = params['carrier_frequency']
    c_ref = params['c_ref']
    theta_diff = 1/2 * (theta_rx - theta)
    delta_d = d_grid.delta
    nd, n_theta = d_grid.shape, theta_diff.size
    pulses_lengths = np.zeros(theta_diff.size)
    for i, theta_d in enumerate(theta_diff):
        a = -(np.pi * carrier_frequency * ratio) ** 2 / (4 * np.log(10 ** (-6 / 20)))
        sigma = c_ref / 2 / np.sqrt(a * np.cos(theta_d))
        nd = np.int(np.floor(2 * n_sigma * sigma / delta_d + 1))
        nd = (nd//2)*2 + 1
        pulses_lengths[i] = nd
    big_pulse_size = int(np.max(pulses_lengths))
    abscisses_pulse = np.linspace(-(big_pulse_size - 1) / 2 * delta_d,
                                  (big_pulse_size - 1) / 2 * delta_d,
                                  big_pulse_size)
    mat_pulse = np.zeros((n_theta, big_pulse_size))
    time_pulse = abscisses_pulse / c_ref * 2 * np.cos(theta_diff)[:, np.newaxis]
    mat_pulse = sp.gausspulse(time_pulse, carrier_frequency, ratio) * np.cos(theta_diff)[:, np.newaxis]
    if debug:
        print('Size of the largest filter : ', big_pulse_size)
        metric = np.std(np.sum(mat_pulse, axis=-1))  # Should be almost 0 to have constant energy
        print('Std des intégrales : ', metric)
        plt.matshow(mat_pulse)
        plt.show()

    def forward(mes):
        return sp.convolve(mes.reshape(nd, n_theta), mat_pulse, 'same').flatten()

    def backward(mes):
        return forward(mes).conj()
    convolution_operator = ssl.LinearOperator(shape=(nd*n_theta, nd*n_theta), matvec=forward, rmatvec=backward,
                                              dtype=float)
    return convolution_operator


if __name__ == '__main__':
    from Donnees.parameters_dico import experiment_parameters
    d_min = -1e-2
    d_max = 1e-1
    lambda_ref = experiment_parameters['c_ref']/experiment_parameters['carrier_frequency']
    d_grid_delta = lambda_ref / 16
    Nd = int((d_max - d_min)/d_grid_delta)
    print('bon delta_d', lambda_ref/16)
    d_grid = grid.Grid1D(Nd, d_grid_delta, d_min)
    N_theta = 40
    theta = 0.8
    theta_rx = np.linspace(-.2, .2, N_theta)
    get_convolution_operator(d_grid, theta_rx, theta, experiment_parameters, n_sigma=4, debug=True)
