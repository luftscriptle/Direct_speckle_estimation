import numpy as np


def o_sel(theta, params):
    """
    :param theta: Array of angles for which we wish to compute the o_sel coefficient
    :param params: Dictionary containing the relevant parameters: probe_elem_width, c_ref, carrier_frequency
    :return:
    """
    w_elem = params['probe_elem_width']
    c_ref = params['c_ref']
    freq = params['carrier_frequency']
    lambda_wave = c_ref/freq
    coefficient = np.cos(theta) * np.sinc(w_elem / lambda_wave * np.sin(theta))
    return coefficient


def get_z(d_grid, theta_rx, x_gamma):
    return d_grid[:, np.newaxis, np.newaxis]/np.cos(theta_rx[np.newaxis, :, np.newaxis]) -\
           x_gamma[np.newaxis, np.newaxis, :]*np.tan(theta_rx[np.newaxis, :, np.newaxis])


def tukey_window(val, params):
    """
    :param val: x-position of the point we wish to compute the windowing
    :param params: dict containing the parameters probe_elem_width, probe_n_elem, probe_pitch
    :param alpha: parameter of the windowing
    :return:
    """

    w_elem = params['probe_elem_width']
    n_elem = params['probe_n_elem']
    pitch = params['probe_pitch']
    alpha = params['tukey_parameter']

    length = (n_elem-1)*pitch + w_elem

    val_1 = (0.5 * (1 - np.cos((2 * np.pi * (val + length/2))/(alpha * length))) *
             np.logical_and(np.less_equal(-length/2, val), np.less(val, -(1 - alpha) * length/2)))

    val_2 = np.logical_and(np.less_equal(-(1 - alpha) * length/2, val), np.less(val, (1 - alpha) * length/2))

    val_3 = (0.5 * (1 - np.cos((2 * np.pi * (val - length/2))/(alpha * length))) *
             np.logical_and(np.less_equal((1 - alpha) * length/2, val), np.less(val, length/2)))

    return val_1 + val_2 + val_3


def o_tx(z_computed, x_gamma, theta_tx, params):
    angle_coeff = o_sel(theta_tx, params)/np.cos(theta_tx)
    arg_w = x_gamma[np.newaxis, np.newaxis, :] - np.tan(theta_tx)*z_computed
    return angle_coeff*tukey_window(arg_w, params)


def o_rx(z_computed, theta_rx, x_gamma, params):
    xt = x_gamma[np.newaxis, np.newaxis, :] - z_computed*np.tan(theta_rx[np.newaxis, :, np.newaxis])
    coefficient = 1/np.sqrt(np.sqrt((x_gamma[np.newaxis, np.newaxis, :]-xt)**2 + z_computed**2))\
            * o_sel(theta_rx, params)[np.newaxis, :, np.newaxis]
    return coefficient


def o_tot(theta_tx, theta_rx, x_gamma, d_grid, params):
    z_computed = get_z(d_grid, theta_rx, x_gamma)
    tx_coeff = o_tx(z_computed, x_gamma, theta_tx, params)
    rx_coeff = o_rx(z_computed, theta_rx, x_gamma, params)
    return tx_coeff * rx_coeff






def get_weight_operator(theta_rx_grid, d_grid, sigma_grid, params):
    x_coord, z_coord = sigma_grid.coordinates
    c_ref = params['c_ref']
    theta_tx = params['angles']
    theta_rx = np.asarray(theta_rx_grid.coordinates)





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Donnees.parameters_dico import  experiment_parameters
    x_gamma = 2 * ((np.arange(experiment_parameters['probe_n_elem']) * experiment_parameters['probe_pitch']) -
       ((experiment_parameters['probe_n_elem'] - 1) / 2 * experiment_parameters['probe_pitch']))
    d_grid = np.linspace(5e-2, 1e-1, 200
                         )
    theta_rx = np.linspace(-0.43, 0.43
                           , 3)
    theta_tx = 0.1
    o_full = o_tot(theta_tx, theta_rx, x_gamma, d_grid, experiment_parameters)
    print(o_full.shape)
    fig, ax = plt.subplots(1, 3)
    [ax[i].matshow(o_full[:, i, :]- o_full[:,1,:]) for i in range(len(ax.ravel()))]
    plt.show()
