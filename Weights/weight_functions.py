import numpy as np
import tqdm
import scipy.sparse
def o_sel(theta, params):
    """
    :param theta: Array of angles for which we wish to compute the o_sel coefficient
    :param params: Dictionary containing the relevant parameters: probe_elem_width,
     c_ref, carrier_frequency
    :return:
    """
    w_elem = params['probe_elem_width']
    c_ref = params['c_ref']
    freq = params['carrier_frequency']
    lambda_wave = c_ref/freq
    coefficient = np.cos(theta) * np.sinc(w_elem / lambda_wave * np.sin(theta))
    return coefficient


def get_z(d_grid, theta_rx, x_gamma):
    return d_grid[:, np.newaxis, np.newaxis]/np.cos(theta_rx[np.newaxis, :, np.newaxis]) \
           - x_gamma[np.newaxis, np.newaxis, :]*np.tan(theta_rx[np.newaxis, :, np.newaxis])


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
             np.logical_and(np.less_equal(-length/2, val),
                            np.less(val, -(1 - alpha) * length/2)))

    val_2 = np.logical_and(np.less_equal(-(1 - alpha) * length/2, val),
                           np.less(val, (1 - alpha) * length/2))

    val_3 = (0.5 * (1 - np.cos((2 * np.pi * (val - length/2))/(alpha * length))) *
             np.logical_and(np.less_equal((1 - alpha) * length/2, val),
                            np.less(val, length/2)))

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


def o_tot(theta_tx, z_computed, theta_rx, x_gamma, params):
    tx_coeff = o_tx(z_computed, x_gamma, theta_tx, params)
    rx_coeff = o_rx(z_computed, theta_rx, x_gamma, params)
    tot_coeff = tx_coeff * rx_coeff
    return tot_coeff


def get_weight_operator(d_grid, theta_rx, sigma_grid, params):
    x_coord, z_coord = sigma_grid.coordinates
    x_offset, z_offset = sigma_grid.offset
    x_delta, z_delta = sigma_grid.delta
    Nd, Nx, Nz, N_theta = d_grid.size, x_coord.size, z_coord.size, theta_rx.size
    theta_tx = params['angle_tx']
    z_computed = get_z(d_grid, theta_rx, x_coord)
    o_full = o_tot(theta_tx, z_computed, theta_rx, x_coord, params)
    z_center = (z_computed - z_offset)/z_delta
    ind_z_inf = np.int32(np.floor(z_center))
    ind_z_sup = ind_z_inf+1
    z_sup = z_offset + ind_z_sup*z_delta
    z_inf = z_offset + ind_z_inf*z_delta
    validity_matrix_inf = np.where(np.logical_and(ind_z_inf >= 0, ind_z_inf < Nz), True, False)

    validity_matrix_sup = np.where(np.logical_and(ind_z_sup >= 0, ind_z_sup < Nz), True, False)
    ind_z_inf = ind_z_inf[validity_matrix_inf]
    ind_z_sup = ind_z_sup[validity_matrix_sup]

    alpha = (z_computed - z_inf) / z_delta
    beta = (z_sup - z_computed) / z_delta
    alpha_val = alpha[validity_matrix_inf]
    beta_val = beta[validity_matrix_sup]
    tic = time.time()
    ind_d = (np.arange(Nd)[:, np.newaxis, np.newaxis]\
        * np.ones((N_theta, Nx), dtype=np.int32))
    ind_x = (np.arange(Nx)*np.ones((Nd, N_theta), dtype=np.int32)[..., np.newaxis])
    ind_theta = (np.arange(N_theta)[np.newaxis, :, np.newaxis]\
        * np.ones((Nd, Nx), dtype=np.int32)[:, np.newaxis, :])
    ind_d_inf = ind_d[validity_matrix_inf]
    ind_d_sup = ind_d[validity_matrix_sup]
    ind_x_inf = ind_x[validity_matrix_inf]
    ind_x_sup = ind_x[validity_matrix_sup]
    ind_theta_inf = ind_theta[validity_matrix_inf]
    ind_theta_sup = ind_theta[validity_matrix_sup]
    ind_d_theta_inf = ind_d_inf*N_theta + ind_theta_inf
    ind_x_z_inf = ind_x_inf*Nz + ind_z_inf
    val_inf = alpha_val*o_full[validity_matrix_inf]
    val_sup = beta_val*o_full[validity_matrix_sup]
    ind_d_theta_sup = ind_d_sup*N_theta + ind_theta_sup
    ind_x_z_sup = ind_x_sup*Nz + ind_z_sup
    ind_col_full = np.concatenate((ind_x_z_inf, ind_x_z_sup))
    ind_row_full = np.concatenate((ind_d_theta_inf, ind_d_theta_sup))
    val_full = np.concatenate((val_inf, val_sup))
    operator = scipy.sparse.csc_matrix((val_full, (ind_row_full, ind_col_full)), shape=(Nd*N_theta, Nx*Nz))
    return operator


# list_ind_x_full = np.stack((list_ind_x,) * Nd * N_theta * 2).flatten()
# list_ind_theta_full = np.array([[i] * Nx for i in range(N_theta)] * Nd * 2).flatten()
# list_ind_d_full = np.array([[i] * Nx * N_theta for i in range(Nd)] * 2).flatten()
# list_ind_z_full = np.stack((ind_z_inf, ind_z_sup)).flatten()
# plt.plot(list_ind_x_full)
# plt.plot(list_ind_d_full)
# plt.plot(list_ind_theta_full)
# plt.show()

# print('size of z_ind_full : ', list_ind_z_full.size)
# print('size of x_ind_full : ', list_ind_x_full.size)
# print('Nx = ', Nx)
# print('max of x_ind : ', list_ind_x_full.max())
# print('size of theta_ind_full : ', list_ind_theta_full.size)
# print('N_theta = ', N_theta)
# print('max of theta_ind : ', list_ind_theta_full.max())
# print('size of d_ind_full : ', list_ind_d_full.size)
# print('Nd = ', Nd)
# print('max of d_ind : ', list_ind_d_full.max())


list_row_ind = []
list_col_ind = []
value = []
# print('expected size : ', Nx*Nd*N_theta*2)
# print('size of list_ind_x_full : ', list_ind_x_full)
# print('size of list_ind_theta_full : ', list_ind_theta_full)
# print('size of list_ind_d_full : ', list_ind_d_full)
# plt.plot(list_ind_z_full)
# plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Donnees.parameters_dico import  experiment_parameters
    import Samuel.core.grid as grid
    import time
    sigma_ref = np.log(
        np.load('C:\\Users\\louis\\Documents\\ARPE\\Projet-ARPE\\projet_arpe\\code\\'
                'Optimisation\\sigma_rmse.npy'))
    delta_x_grid = 2 * experiment_parameters['probe_pitch']
    delta_z_grid = 2 * experiment_parameters['probe_pitch']
    experiment_parameters['angle_tx'] = 0.1
    Nx = 110
    Nz = 110
    z_min = 5e-3
    x_min = -Nx * delta_x_grid / 2
    sigma_grid = grid.GridND((Nx, Nz), (delta_x_grid, delta_z_grid), (x_min, z_min))
    x_coord, z_coord = sigma_grid.coordinates
    print(z_coord)
    x_gamma = 2 * ((np.arange(experiment_parameters['probe_n_elem']) * experiment_parameters['probe_pitch']) -
       ((experiment_parameters['probe_n_elem'] - 1) / 2 * experiment_parameters['probe_pitch']))
    d_grid = np.linspace(5e-3, 3e-2, 40)
    theta_rx = np.linspace(-0.1, 0.1, 70)
    operator = get_weight_operator(d_grid, theta_rx, sigma_grid, experiment_parameters)
    mes = np.load('C:\\Users\\louis\\Documents\\ARPE\\Projet-ARPE\\projet_arpe\\code\\Donnees_pwave\\data.npy')
    print(mes.shape)