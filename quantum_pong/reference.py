
"""
http://www.physics.buffalo.edu/phy410-505-2009/topic5/lec-5-5.pdf
simulation module

todo; allow for multiple inteacting electons
"""
import numpy as np
import matplotlib.colors

# global constants
h_bar = 1
mass = 1e-3
E = 1


def forward(x):
    return np.fft.fft2(x)

def backward(x):
    return np.fft.ifft2(x)

def fftfreq(shape):
    """grid of k vectors for a given shape"""
    ndim = len(shape)
    f = np.empty(tuple(shape) + (ndim,), np.float32)
    for i, s in enumerate(shape):
        q = [1] * ndim
        q[i] = s
        f[..., i] = np.fft.fftfreq(s).reshape(q)
    return f

def density(phi):
    return np.abs(phi) ** 2

def phase(phi):
    return np.angle(phi)

def phi_to_hsv(phi, V=None):
    d = density(phi)
    d /= d.max()
    d = np.sqrt(d)
    p = phase(phi)
    p = (p + np.pi) / (2 * np.pi)
    ones = np.ones_like(d)
    if V is None:
        V = np.zeros_like(d)
    E = d + V
    hsv = np.dstack([p, ones, d])
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb


def create_grid(shape, dimensions):
    """construt coordinates grid and corresponding spatial frequency vectors"""
    shape = np.asarray(shape)
    dimensions = np.asarray(dimensions)
    ndim = len(shape)
    coordinates = np.indices(shape, np.float32)
    coordinates = np.moveaxis(coordinates, 0, ndim)
    coordinates = coordinates - (shape - 1) * 0.5
    coordinates = coordinates / shape * dimensions
    spacing = dimensions / shape
    frequencies = fftfreq(shape)
    return coordinates, frequencies


class Simulation(object):

    def __init__(self, shape, dimensions, tau, V=None):
        self.shape = shape
        self.dimensions = dimensions
        self.coordinates, self.wave_vectors = create_grid(shape, dimensions)

        self.phi = np.zeros(shape, np.complex64)

        if V is None:
            V = np.zeros(shape, np.float32)
            V = ((self.coordinates * 3) ** 2).sum(axis=-1) * 200
        self.V = V

        self.set_timestep(tau)

    def set_timestep(self, tau):
        self.tau = tau
        self.update_steps()

    def update_steps(self):
        # the amount of complex rotation to impart in position and momentum space
        self.T_step = (self.wave_vectors ** 2).sum(axis=-1) / mass / h_bar * self.tau / 2
        self.V_step = self.V / 2 / h_bar * self.tau / 2

        self.T_step = np.exp(self.T_step * 1j)
        self.V_step = np.exp(self.V_step * 1j)

    def timestep(self):
        self.phi = backward(forward(self.phi * self.V_step) * self.T_step) * self.V_step
        return self.phi

    def get_image(self):
        return phi_to_hsv(self.phi, self.V)

    def gaussian_wave(self, pos, vec, sigma):
        r = self.coordinates - pos
        norm = np.sqrt(sigma * np.sqrt(np.pi))
        gaussian = np.exp(-((r ** 2).sum(axis=-1) / sigma ** 2)) / norm
        # k_0 = np.sqrt(2 * mass * E - h_bar ** 2 / 2 / sigma ** 2) / h_bar
        wave = np.exp(self.coordinates.dot(vec) * 1j)
        return gaussian * wave


if __name__ == '__main__':
    shape = 2**8, 2**8

    sim = Simulation(shape, [1, 1], tau=0.1)
    sim.phi = sim.gaussian_wave(pos=[0.0, 0.0], vec=[100, 30], sigma=0.1)

    import matplotlib.pyplot as plt
    plt.imshow(sim.get_image())
    plt.show()
