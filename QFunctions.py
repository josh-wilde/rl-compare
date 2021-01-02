# Set up linear q function approximation here
import numpy as np

class LinearTilingQApproximator():
    def __init__(self, iht, n_tilings, tile_resolution):
        # index hash table that stores the tilings
        self.iht = iht
        self.n_tilings = n_tilings
        self.tile_resolution = tile_resolution

        # Initialize weights to zero
        self.w = [0] * self.iht.size

    def get_active_tiles(self, state, action):
        assert len(state) <= 3
        assert len(state) >= 2
        if len(state) >= 2:
            state_scaled = [self.tile_resolution * state[0] / (0.5+1.2),
                            self.tile_resolution * state[1] / (0.07+0.07)]
        if len(state) == 3:
            state_scaled = state_scaled + [self.tile_resolution * state[2] / (0.14+0.14)]
            
        return tiles(self.iht, self.n_tilings, state_scaled, [action])

    def __call__(self, state, action):
        active_tiles = self.get_active_tiles(state, action)
        return sum([w[t] for t in active_tiles])

    def update(self, state, action, alpha, error):
        active_tiles = self.get_active_tiles(state, action)

        for t in active_tiles:
            self.w[t] = self.w[t] + alpha * error
