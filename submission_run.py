import torch
import numpy as np


def list_all_values(lis):
    result = []
    for dic in lis:
        for val in dic.values():
            result.append(val)
    return result


class PredatorAgent:
    def __init__(self, path="models/predator119.pt"):
        self.model = torch.load(path, map_location="cpu")
        self.model.eval()

    def act(self, state_dict):
        res = np.zeros(2)
        for i in range(2):
            state = list(state_dict['predators'][i].values()) + \
                    list_all_values(state_dict['preys'])
            res[i] = self.model(torch.Tensor(state)).item()
        return res


class PreyAgent:
    def __init__(self, path="models/prey119.pt"):
        self.model = torch.load(path, map_location="cpu")
        self.model.eval()

    def act(self, state_dict):
        res = np.zeros(5)
        for i in range(5):
            state = list_all_values(state_dict['predators']) + \
                    list(state_dict['preys'][i].values())
            res[i] = self.model(torch.Tensor(state)).item()
        return res


if __name__ == "__main__":
    from predators_and_preys_env.env import PredatorsAndPreysEnv

    predator = PredatorAgent()
    prey = PreyAgent()
    env = PredatorsAndPreysEnv(render=True)

    for _ in range(25):
        state = env.reset()
        done = False
        while not done:
            pred_actions = predator.act(state)
            prey_actions = prey.act(state)
            state, done = env.step(pred_actions, prey_actions)
