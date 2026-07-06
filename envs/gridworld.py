import numpy as np


class GridWorld:
    def __init__(self, size=10):    
        self.size = size
        self.reset()

    def reset(self):
        self.agent = np.array([0, 0], dtype=np.float32)
        return self.agent.copy()

    def step(self, action):
        x, y = self.agent

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.size - 1:
            y += 1

        self.agent = np.array([x, y], dtype=np.float32)
        return self.agent.copy()
