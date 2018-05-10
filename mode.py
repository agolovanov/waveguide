import numpy as _np

class Mode:
    """
    Constructs a mode with the longitudinal wavenumber h based on the Hertz and the magnetic Hertz vectors
    """
    def __init__(self, h, hertz, hertz_dx, hertz_dy, m_hertz, m_hertz_dx, m_hertz_dy):
        self.h = h
        self.__kappa_sq = 1 - h * h
        self.hertz = hertz
        self.hertz_dx = hertz_dx
        self.hertz_dy = hertz_dy
        self.m_hertz = m_hertz
        self.m_hertz_dx = m_hertz_dx
        self.m_hertz_dy = m_hertz_dy

    def __long_function(self, z):
        return _np.exp(1j * self.h * z)

    def ex(self, x, y, z=0):
        return (self.h * self.hertz_dx(x, y) + self.m_hertz_dy(x, y)) * self.__long_function(z)

    def ey(self, x, y, z=0):
        return (self.h * self.hertz_dy(x, y) - self.m_hertz_dx(x, y)) * self.__long_function(z)

    def ez(self, x, y, z=0):
        return -1j * self.__kappa_sq * self.hertz(x, y) * self.__long_function(z)

    def hx(self, x, y, z=0):
        return (- self.hertz_dy(x, y) + self.h * self.m_hertz_dx(x, y)) * self.__long_function(z)

    def hy(self, x, y, z=0):
        return (self.hertz_dx(x, y) + self.h * self.m_hertz_dy(x, y)) * self.__long_function(z)

    def hz(self, x, y, z=0):
        return -1j * self.__kappa_sq * self.m_hertz(x, y) * self.__long_function(z)


class ModeTM(Mode):
    def __init__(self, h, hertz, hertz_dx, hertz_dy):
        empty = lambda x, y: _np.zeros(x.shape)
        Mode.__init__(self, h, hertz, hertz_dx, hertz_dy, empty, empty, empty)


class ModeTE(Mode):
    def __init__(self, h, hertz, hertz_dx, hertz_dy):
        empty = lambda x, y: _np.zeros(x.shape)
        Mode.__init__(self, h, empty, empty, empty, hertz, hertz_dx, hertz_dy)