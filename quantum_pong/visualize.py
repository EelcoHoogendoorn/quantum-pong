"""
quantum-pong ui
"""

import numpy as np
import vispy.app
from vispy import gloo
from vispy import visuals
from vispy.visuals.transforms import STTransform

from reference import Simulation

class Canvas(vispy.app.Canvas):
    def __init__(self, simulation):
        self.simulation = simulation
        vispy.app.Canvas.__init__(self, keys='interactive', size=(800, 800))
        self.image = visuals.ImageVisual(self.simulation.get_image(), method='subdivide')

        # scale and center image in canvas
        s = 700. / max(self.image.size)
        t = 0.5 * (700. - (self.image.size[0] * s)) + 50
        self.image.transform = STTransform(scale=(s, s), translate=(t, 50))

        self.show()

        self._timer = vispy.app.Timer('auto', connect=self.update, start=True)

    def on_draw(self, ev):
        gloo.clear(color='black', depth=True)
        self.image.set_data(self.simulation.get_image())
        self.image.draw()
        self.simulation.timestep()
        print('step')

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.image.transforms.configure(canvas=self, viewport=vp)



if __name__ == '__main__':
    sim = Simulation((256, 256), [1, 1], 0.1)
    sim.phi = sim.gaussian_wave(pos=[0.0, -0.0], vec=[200, 50], sigma=0.05)

    win = Canvas(sim)
    # import sys
    # if sys.flags.interactive != 1:
    vispy.app.run()
