if not 'np' in dir():
    import numpy as np


def grid_sigma(nx, nz, x_min, x_max, z_min, z_max):
    x_grid = np.linspace(x_min, x_max, nx)
    z_grid = np.linspace(z_min, z_max, nz)
    xx, zz = np.meshgrid(x_grid, z_grid)
    sigma_ref = np.where((xx - 1.2e-2)**2 + (zz - 3e-2)**2 <= 7e-3**2, 4, 0)\
                + np.where((xx + 1.2e-2)**2 + (zz - 2e-2)**2 <= 2.5e-3**2, 16, 0)\
                + np.where((xx + 1.2e-2)**2 + (zz - 4e-2)**2 <= 2.5e-3**2, 16, 0)
    return sigma_ref.T, x_grid, z_grid

def grid_sigma_from_grid_object(grid):
    """

    :param grid: objet grid, doit etre tq grid.ndim = 2
    :return:
    """
    if grid.n_dim != 2:
        raise TypeError('Expected a 2D grid, got {} dimmension(s) instead'.format(grid.n_dim))
    x_grid, z_grid = grid.coordinates
    xx, zz = np.meshgrid(x_grid, z_grid)
    sigma_ref = np.where((xx - 1.2e-2) ** 2 + (zz - 3e-2) ** 2 <= 7e-3 ** 2, 4, 0) \
                + np.where((xx + 1.2e-2) ** 2 + (zz - 2e-2) ** 2 <= 2.5e-3 ** 2, 16, 0) \
                + np.where((xx + 1.2e-2) ** 2 + (zz - 4e-2) ** 2 <= 2.5e-3 ** 2, 16, 0)
    return sigma_ref.T
