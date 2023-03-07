from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results

import numpy as np

import pickle


class TPGAgent(BaseAgent):
    def __init__(self, name="blue_agent_0", np_random=None, path="tpg/team_file"):
        super().__init__(name, np_random)

        self.team = None
        with open(path, "rb") as file:
            self.team = pickle.load(file)
        self.memMatrix = np.zeros((100,8))
        
    def get_action(self, observation, action_space):
        act = self.team.act(observation, self.memMatrix)
        return int(abs(act[1][0])) % action_space.n
