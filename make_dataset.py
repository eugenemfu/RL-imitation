import torch
from torch.utils.data import TensorDataset

from baseline import ChasingPredatorAgent, FleeingPreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv


def list_all_values(lis):
    result = []
    for dic in lis:
        for val in dic.values():
            result.append(val)
    return result


MAX_SIZE = 10000000
SAVE_SIZE = 10000
STEP = 5
OUTPUT = 'dataset.pkl'

env = PredatorsAndPreysEnv()
done = True

predator = ChasingPredatorAgent()
prey = FleeingPreyAgent()

states = []
actions = []

last_save_size = 0

for i in range(MAX_SIZE * STEP):

    if done:
        state = env.reset()
        done = False

    predator_action = predator.act(state)
    prey_action = prey.act(state)
    next_state, done = env.step(predator_action, prey_action)

    if i % STEP == 0:
        states.append(list_all_values(state['predators']) +
                      list_all_values(state['preys']))
        actions.append(predator_action + prey_action)

    print(f"\rIterations: {i+1:<8d}  dataset size: {len(states):<8d}", end='')

    if len(states) % SAVE_SIZE == 0 and len(states) != last_save_size:
        dataset = TensorDataset(torch.Tensor(states), torch.Tensor(actions))
        #print(len(dataset))
        torch.save(dataset, OUTPUT)
        print('  - saved to', OUTPUT)
        last_save_size = len(states)

    state = next_state

    if len(states) == MAX_SIZE:
        break
