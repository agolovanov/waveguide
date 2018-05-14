import mode as _mode
import numpy as _np


def r(x, y):
    return _np.sqrt(x ** 2 + y ** 2)


def phi(x, y):
    return _np.arctan2(y, x)


class RoundWaveguide:
    def __init__(self, radius):
        self.radius = radius

    def __get_mode(self, m, n, phi0, kappa, is_TE=True):
        from scipy.special import jn, jvp
        h = _np.sqrt(1 - kappa ** 2)

        def radial_f(r):
            return _np.where(r < self.radius, jn(m, kappa * r), 0.0)

        def radial_f_der(r):
            return _np.where(r < self.radius, jvp(m, kappa * r), 0.0)

        def hertz(x, y):
            return radial_f(r(x, y)) * _np.cos(m * (phi(x, y) - phi0))

        def hertz_dx(x, y):
            r_v = r(x, y)
            phi_v = phi(x, y) - phi0
            return kappa * radial_f_der(r_v) * _np.cos(m * phi_v) * x / r_v + \
                   radial_f(r_v) * m * _np.sin(m * phi_v) * y / r_v ** 2

        def hertz_dy(x, y):
            r_v = r(x, y)
            phi_v = phi(x, y) - phi0
            return kappa * radial_f_der(r_v) * _np.cos(m * phi_v) * y / r_v - \
                   radial_f(r_v) * m * _np.sin(m * phi_v) * x / r_v ** 2

        if is_TE:
            return _mode.ModeTE(h, hertz, hertz_dx, hertz_dy)
        else:
            return _mode.ModeTM(h, hertz, hertz_dx, hertz_dy)

    def get_TE_mode(self, m, n, phi0=0):
        from scipy.special import jnp_zeros
        mu = jnp_zeros(m, n)[-1]
        kappa = mu / self.radius
        return self.__get_mode(m, n, phi0, kappa, True)

    def get_TM_mode(self, m, n, phi0=0):
        from scipy.special import jn_zeros
        nu = jn_zeros(m, n)[-1]
        kappa = nu / self.radius
        return self.__get_mode(m, n, phi0, kappa, False)