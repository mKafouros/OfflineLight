import numpy as np
from . import BaseGenerator

class TravelTimeGenerator(BaseGenerator):

    def __init__(self, world, I, negative=False):
        self.world = world
        self.I = I


        self.ob_length = 1
        self.negative = negative

    def ob_space(self):
        return np.array([self.ob_length,])

    def generate(self):
        travel_time = self.world.eng.get_average_travel_time()
        if self.negative:
            travel_time *= -1
        return travel_time
