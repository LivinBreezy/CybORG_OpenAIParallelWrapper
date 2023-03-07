# CybORG (Cage 3) Open AI Gym Parallel Wrapper Repository

This repository contains the current code for the  the Cage 3 OpenAI Gym Parallel wrapper.

The CybORG Cage Challenge 3 OpenAIGymWrapper is currently designed to provide a flat 
observation space and discrete action space in order to conform to the OpenAI Gym
standard. However, one drawback of the implementation is that it must be initialized to
handle a single drone at a time, which makes the environment very sparse. Currently,
it produces an action space of 56 (for a single drone), with an observation space of 11,293.

## Motivation

Since we use TPG, and our algorithms have been designed to work with OpenAI Gym in the past,
it seemed like a good place to start. For Cage Challenge 2, this appeared to be sufficient
as there was a single blue agent to control by default. In Cage 3, the environment requires
18 agents (or at least 18 generated actions) in a single step, which means we need to adjust
the way the system is presented in order to best allow TPG to handle it.

For this iteration of this wrapper, we have opted to allow OpenAIGymWrapper to take advantage
of the newer CybORG 3.0+ features. One of the major changes between Cage 2 and Cage 3 is the
ability for CybORG to have a "parallel step", which allows agents to apply multiple actions
to the same simulation step; currently, the OpenAIGymWrapper only allows a single action to be
applied per step, which means every time an agent takes an action, the entire red team gets 
to apply theirs. By converting the OpenAIGymWrapper to handle the parallel action step, we
can submit an entire drone swarm's worth of actions, as intended by the newest version of CybORG.

## Implementation

The current implementation begins with the standard base for OpenAIGymWrapper, which is both
the OpenAIGymWrapper itself, but also the FixedFlatWrapper. The OpenAI wrapper actually comes
in two parts in order to achieve the desired observation and action spaces:

1. The FixedFlatWrapper is applied to the environment first, which provides functions for 
   "flattening" the otherwise categorical observation space. From a simple reading, it appears
   to go to each observation category and convert each piece of data into a floating point number.
   It appears to do this across a universal observation space (all the drones put together), but
   is made partially observable in the process based on the current drone perspective. That is,
   you ask it for a specific drone's observation and it produces a global observation set with
   only the requested drone's observable space inserted. The remaining features are set to -1.0.
 
2. The OpenAIGymWrapper is applied to the FixedFlatWrapper environment (making CybORG technically
   double wrapped) and provides the discrete action space conversion. This means the OpenAIGymWrapper
   uses its own action space and the FixedFlatWrapper observation space to provide to the agent.
   In its current implementation, it still uses the CybORG 2.0+ step function, which is marked
   as a compatibility function for Cage Challenges 1 and 2.

Our new wrappers, *FixedFlatParallelWrapper* and *OpenAIGymParallelWrapper* are based on these 
wrappers, with a combination of some updated features from the PettingZoo wrapper, which has been
updated to include the parallel step. These wrappers are found in `CybORG/Agents/Wrappers`.

The FixedFlatParallelWrapper contains the following modifications:

### get_observation(agent: str, change: bool)

This function has been updated to contain a new parameter, `change`, which allows a user to control
whether or not an observation by agent name is returned already flatted or left alone. This is done
to have tighter control over which observations are returned at certain points in the parallel-compatibility
process. Currently is not used with change-mode disabled.

### parallel_step(actions:dict, messages: dict)

This function has been added as a compatibility step between environments. If it is necessary to call
an underlying env.parallel_step(..), this can be used as a translation function.

The OpenAIGymParallelWrapper contains the following modifications:

### __init___(env: BaseWrapper)

The OpenAIGymParallelWrapper constructor has been altered and moves away from storing single actions
and single agents. Instead, it creates and maintains both the action space and observation space for 
a collection of agents, rather than a single agent.

### step(actions: dict)

The new step function follows the same rules as the previous step function, except rather than converting
a single action and storing it as a class member, it converts a dictionary of actions tied to agent->action
pairs. Once the actions are converted, it calls the env.parallel_step(..) function on said actions and receives
a series of "raw" observations (not yet changed by the FixedFlatParallelWrapper), along with the other 
standard state variables. The function then converts each observation received into a flat observation space
and returns them as a dictionary.

This function will require some modifications in the future, discussed below in the `Improvements` section.

### reset(..)

The reset function is similar to previous, except it now contains some of the modifications of the step
function so it may handle a starting reward and observation across multiple agents.

### Additional Functions

Additional properties and functions were migrated from the PettingZoo wrapper in order to better support 
the evaluation programs, which require functions such as `get_dones()` and `action_spaces()`.

## Analysis

The wrapper is currently working as intended, although it slightly underperforms compared to the PettingZoo
wrapper. I believe the biggest impact on the performance is a combination of Python itself and the `observation_change`
functions, which are used in each wrapper to convert the observation space to match a wrapper's standard.

The tests below are samples from basic runtime speed comparisons between OpenAIGymParallelWrapper and the 
PettingZooParallelWrapper. It should be noted that the PettingZoo wrapper does not need to produce the universe
of features (only a single drone's) each time, while the OpenAI Gym wrappers do. These blocks show the step time
taken to process an entire step function for the current environment. That means it does not include agent action
generation time. The times below are not adjusted to account for the impact of the timing itself.

All times below are in nanoseconds (ns).

### PettingZooParallelWrapper Observation Change and Step Times

```
1  Parallel ObsChange Time: 0
2  Parallel ObsChange Time: 975800
3  Parallel ObsChange Time: 0
4  Parallel ObsChange Time: 0
5  Parallel ObsChange Time: 0
6  Parallel ObsChange Time: 0
7  Parallel ObsChange Time: 0
8  Parallel ObsChange Time: 0
9  Parallel ObsChange Time: 0
10 Parallel ObsChange Time: 0
11 Parallel ObsChange Time: 0
12 Parallel ObsChange Time: 975900
13 Parallel ObsChange Time: 0
15 Parallel ObsChange Time: 0
16 Parallel ObsChange Time: 0
17 Parallel ObsChange Time: 0
18 Parallel ObsChange Time: 0

TOTAL STEP TIME: 142495900
```

The PettingZoo wrapper converts the observation space quite quickly, usually with only a few timings registering
a longer time gap. In most cases, the system appears to detect the time taken as 0.

### OpenAIGymParallelWrapper Observation Change and Step Times

```
1  OAIParallel ObsChange Time: 976200
2  OAIParallel ObsChange Time: 975900
3  OAIParallel ObsChange Time: 1952100
4  OAIParallel ObsChange Time: 976100
5  OAIParallel ObsChange Time: 1952100
6  OAIParallel ObsChange Time: 1952000
7  OAIParallel ObsChange Time: 976000
8  OAIParallel ObsChange Time: 976100
9  OAIParallel ObsChange Time: 975800
10 OAIParallel ObsChange Time: 975900
11 OAIParallel ObsChange Time: 976400
12 OAIParallel ObsChange Time: 1952200
13 OAIParallel ObsChange Time: 975900
14 OAIParallel ObsChange Time: 1952100
15 OAIParallel ObsChange Time: 976700
16 OAIParallel ObsChange Time: 1952100
17 OAIParallel ObsChange Time: 975500
18 OAIParallel ObsChange Time: 1951900

TOTAL STEP TIME: 159088000
```

This show an example of a "normal" step timing for the new OpenAIGymParallelWrapper, which appears to be competitive
with PettingZoo. This has a time difference around 10%. However, the OpenAIGymWrapper's (FixedFlatWrapper) original
`observation_change` function is unchanged and can occasionally cause the step time to increase quite dramatically,
as it was likely never intended to be used multiple times per step.

### OpenAIGymParallelWrapper Observation Change and Step Times (Worst Case)

```
1  OAIParallel ObsChange Time: 976000
2  OAIParallel ObsChange Time: 976600
3  OAIParallel ObsChange Time: 1952000
4  OAIParallel ObsChange Time: 1952400
5  OAIParallel ObsChange Time: 1951900
6  OAIParallel ObsChange Time: 976100
7  OAIParallel ObsChange Time: 975600
8  OAIParallel ObsChange Time: 1952400
9  OAIParallel ObsChange Time: 40016100
10 OAIParallel ObsChange Time: 1952300
11 OAIParallel ObsChange Time: 976200
12 OAIParallel ObsChange Time: 1952000
13 OAIParallel ObsChange Time: 976500
14 OAIParallel ObsChange Time: 975900
15 OAIParallel ObsChange Time: 1952000
16 OAIParallel ObsChange Time: 1952100
17 OAIParallel ObsChange Time: 976600
18 OAIParallel ObsChange Time: 1952200

TOTAL STEP TIME: 218623800
```

Although on an average case the timing appears to bounce between ~97000 and ~1950000 ns, occasionally it will take
significantly longer, reaching as high as 40000000 ns (0.04 s) per `observation_change` function call. This is 412
times longer than the fastest steps. This can increase a total step time by upwards of 37%.

## Improvements

I believe there are several ways to improve performance, one of which is more immediately viable.

1. A refactor/rewrite of the `FixedFlatWrapper.observation_change()` function so that it's not as long would be
   the best case scenario, although this may not be viable considering the complexity of the environment
   and the translation process to the flat observation space.
   
2. Writing a custom `observation_change` function for the OpenAIGymParallelWrapper is the most accessible option.
   Instead of generating an entire flat observation set every time a new observation is given by CybORG, it may
   be simpler to instead update the existing feature set with only the changed features. Most of the feature set
   is hidden due to partial observability, and an algorithm to uupdate only changed values could not only be
   useful here, but in other wrappers as well.
   
## Evaluation

Currently the OpenAIGymParallelWrapper works with the evaluation system, including the submission.py and evaluation.py
files, although the evaluation.py file needs to have the first `a` append (for writing to a file) enabled with the second
commented out. It's likely that this could be fixed with an `if` statement.

The line which attempts to retrieve `action_spaces` from `wrapped_cyborg` appears to be missing `()`, so they were added.

An OpenAIGymParallelWrapper evaluation example is included in `CybORG/Agents/Evaluation` with a simple TPG team.