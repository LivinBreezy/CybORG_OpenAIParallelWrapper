import inspect

import numpy as np
from gym import spaces
from typing import Union, List, Optional, Tuple

from prettytable import PrettyTable

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper



class OpenAIGymParallelWrapper(BaseWrapper):
    def __init__(self, env: BaseWrapper):
        super().__init__(env)

        # Initialize the action signature dictionary.
        self.action_signature = {}

        # Initialize the action-related dictionaries.
        self.possible_actions = {}
        self._action_spaces = {}

        # Create a dictionary of action spaces for every drone in Discrete (Gym) format
        # and ensure the possible_actions dictionary is properly populated.
        for agent in self.possible_agents:
            a_space = spaces.Discrete(self.get_action_space(agent))
            self.possible_actions[agent] = self.last_possible_actions
            self._action_spaces[agent] = a_space
    
        # Initialize the observation spaces dictionary.
        self._observation_spaces = {}

        # Retrieve the observation space for each drone, convert it to Gym, then box it.
        for agent in self.possible_agents:
            box_len = len(self.observation_change(agent, self.env.get_observation(agent)))
            self._observation_spaces[agent] = spaces.Box(-1.0, 3.0, shape=(box_len,), dtype=np.float32)

        # Initialize the done flags.
        self.dones = {agent: False for agent in self.possible_agents}

    def step(self, actions: dict = None):
        # Convert all agent actions from discrete to CybORG actions.
        converted_actions = {}
        for agent, action in actions.items():
            if action is not None:
                converted_actions[agent] = self.possible_actions[agent][action]
       
        # Run a parallel step with all of the agents' actions.
        raw_obs, rews, dones, infos = self.env.parallel_step(converted_actions, messages={})

        # Convert every observation from CybORG to Flat.
        observations = {agent: np.array(self.env.observation_change(agent, agent_obs), dtype=np.float32)
                        for agent, agent_obs in raw_obs.items()}
        
        # Save the dones in this object.
        self.dones.update(dones)

        # Create the reward dictionary for all agents. Store it in the object.
        self.rewards = {agent: float(sum(agent_rew.values())) for agent, agent_rew in rews.items()}

        return observations, self.rewards, dones, infos

    @property
    def np_random(self):
        return self.env.get_attr('np_random')

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # Perform a full simulation reset.
        result = self.env.reset()

        # Reset the current rewards.
        self.rewards = {agent: 0.0 for agent in self.possible_agents}

        # Convert every observation from CybORG to Flat.
        observations = {agent: np.array(self.env.get_observation(agent), dtype=np.float32)
                        for agent in self.possible_agents}

         # Re-initialize the action-related dictionaries.
        self.possible_actions = {}
        self._action_spaces = {}

        # Re-create the discrete action space and possible_actions.
        for agent in self.possible_agents:
            a_space = spaces.Discrete(self.get_action_space(agent))
            self.possible_actions[agent] = self.last_possible_actions
            self._action_spaces[agent] = a_space

        if return_info:
            return observations, {}
        else:
            return observations

    def render(self, mode='human'):
        return self.env.render(mode)

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        observation = self.env.get_observation(agent)
        observation = self.observation_change(self.agent_name, observation)
        return np.array(observation, dtype=np.float32)

    def get_agent_state(self,agent:str):
        return self.get_attr('get_agent_state')(agent)

    def get_action_space(self, agent):
        return self.action_space_change(self.env.get_action_space(agent))

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below return the action space as a dict "
        possible_actions = []
        temp = {}
        params = ['action']
        # for action in action_space['action']:
        for i, action in enumerate(action_space['action']):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))

        self.last_possible_actions = possible_actions
        return len(possible_actions)

    @property
    def observation_spaces(self):
        '''
        Returns the observation space for every possible agent
        '''
        try:
            return self._observation_spaces
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead"
            )
    @property
    def action_spaces(self):
        '''
        Returns the action space for every possible agent
        '''
        try:
            return self._action_spaces
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            )

    def get_rewards(self):
        '''
        Returns the rewards for every possible agent
        '''
        try:
            return {agent: self.get_reward(agent) for agent in self.possible_agents}
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            )

    def get_dones(self):
        '''
        Returns the dones for every possible agent
        '''
        try:
            return {agent: self.get_done(agent) for agent in self.possible_agents}
        except AttributeError:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            )

    @property
    def agents(self):
        return [agent for agent in self.env.active_agents if not self.dones[agent]]

    @property
    def possible_agents(self):
        return self.env.agents
