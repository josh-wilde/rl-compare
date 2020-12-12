
class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        '''
        Generally the agent's action should depend on the observation, but here just random
        '''
        return self.action_space.sample()
