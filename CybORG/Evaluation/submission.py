from CybORG.Agents.Wrappers.OpenAIGymParallelWrapper import OpenAIGymParallelWrapper
from CybORG.Agents.Wrappers.FixedFlatParallelWrapper import FixedFlatParallelWrapper

from TPGAgent import TPGAgent

agents = {f"blue_agent_{agent}": TPGAgent() for agent in range(18)}

def wrap(env):
    return OpenAIGymParallelWrapper(env=FixedFlatParallelWrapper(env))

submission_name = 'tpg'
submission_team = 'robert'
submission_technique = 'Tangled Program Graphs'
