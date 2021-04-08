import numpy as np


def o_sel(theta, params):
    """
    :param theta: Array of angles for wich we wish to compute the o_sel coefficient
    :param params: Dictionnary containing the relevant parameters: probe_elem_width, c_ref, carrier_frequency
    :return:
    """
    w_elem = params['probe_elem_width']
    c_ref = params['c_ref']
    freq = params['carrier_frequency']
    lambda_wave = c_ref/freq
    coeff = np.cos(theta) * np.sinc(w_elem / lambda_wave * np.sin(theta))
    return coeff


def get_z(x_gamma, d_grid, theta_rx):
    delta_z = ((x_gamma - d_grid[:,np.newaxis, np.newaxis])*np.sin(theta_rx[np.newaxis, :, np.newaxis]))*np.tan(theta_rx[np.newaxis, :, np.newaxis])
    return d_grid[:, np.newaxis, np.newaxis]*np.cos(theta_rx[np.newaxis, :, np.newaxis]) + delta_z


def o_rx(x_gamma, d, theta_rx, params):
    z_gamma = d[...,np.newaxis]*np.cos(theta_rx) + (x_gamma - d[...,np.newaxis]*np.sin(theta_rx))*np.tan(theta_rx)
    xt = z_gamma*np.tan(theta_rx) + x_gamma
    plt.matshow((x_gamma-xt)**2 + z_gamma**2)
    plt.colorbar()
    plt.show()
    coeff = 1/np.sqrt(np.sqrt((x_gamma-xt)**2 + z_gamma**2))*o_sel(theta_rx, params)
    return coeff


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


def o_tx(theta_tx, theta_rx, x_gamma, d, params):
    angle_coeff = o_sel(theta_tx, params)/np.cos(theta_tx)
    arg_w = x_gamma - np.tan(theta_tx)*get_z(x_gamma, d, theta_rx)
    return angle_coeff*tukey_window(arg_w, params)


def get_single_matrix(theta_rx, d, sigma_grid, params):
    x_coord, z_coord = sigma_grid.coordinates
    c_ref = params['c_ref']
    theta_tx = params['angles']
    theta_mid = 1 / 2 * (theta_tx + theta_rx)
    xx, zz = np.meshgrid(x_coord, z_coord)


def get_weight_operator(theta_rx_grid, d_grid, sigma_grid, params):
    x_coord, z_coord = sigma_grid.coordinates
    c_ref = params['c_ref']
    theta_tx = params['angles']
    theta_rx = np.asarray(theta_rx_grid.coordinates)
    theta_mid = 1/2 * (theta_tx + theta_rx)
    # TODO


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Donnees.parameters_dico import experiment_parameters
    x_gamma = 2 * ((np.arange(experiment_parameters['probe_n_elem']) * experiment_parameters['probe_pitch']) -
       ((experiment_parameters['probe_n_elem'] - 1) / 2 * experiment_parameters['probe_pitch']))
    d_grid = np.linspace(5e-3, 1e-1, 200
                         )
    theta_rx = np.linspace(-0.1, .11, 10)
    z = get_z(x_gamma, d_grid, theta_rx)
    print(z.shape)
    print(z[0].T.shape)
    plt.plot(x_gamma, z[0].T, label=np.round(theta_rx, 3))
    plt.legend()
    plt.show()
