import numpy as np
from weight_functions import get_weight_operator

import matplotlib.pyplot as plt
from Donnees.parameters_dico import experiment_parameters
import Samuel.core.grid as grid
import time
sigma_ref = np.log(
    np.load('C:\\Users\\louis\\Documents\\ARPE\\Projet-ARPE\\projet_arpe\\code\\'
            'Optimisation\\sigma_rmse.npy'))
delta_x_grid = 2 * experiment_parameters['probe_pitch']
delta_z_grid = 2 * experiment_parameters['probe_pitch']
experiment_parameters['angle_tx'] = 0.
Nx = 110
Nz = 110
z_min = 5e-3
x_min = -Nx * delta_x_grid / 2
sigma_grid = grid.GridND((Nx, Nz), (delta_x_grid, delta_z_grid), (x_min, z_min))
x_coord, z_coord = sigma_grid.coordinates
print(z_coord)
x_gamma = 2 * ((np.arange(experiment_parameters['probe_n_elem']) * experiment_parameters['probe_pitch']) -
   ((experiment_parameters['probe_n_elem'] - 1) / 2 * experiment_parameters['probe_pitch']))
Nd, N_theta = 80, 70
d_grid = np.linspace(5e-3, 2e-2, Nd)
theta_rx = np.linspace(-0.1, 0.1, N_theta)
sigma_shape = (Nd, N_theta)

operator = get_weight_operator(d_grid, theta_rx, sigma_grid, experiment_parameters)
