import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predator = nn.Sequential(
    nn.Linear(29, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Tanh(),
).to(device)

prey = nn.Sequential(
    nn.Linear(13, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Tanh(),
).to(device)

NUM_EPOCH = 500
BATCH_SIZE = 2 ** 14
SAVE_EVERY = 1
LR = 1e-4
DATA = 'dataset.pkl'

dataset_train = torch.load(DATA)
dataset_test, dataset_train = torch.utils.data.random_split(
    dataset_train, [100000, len(dataset_train) - 100000])

# print(len(dataset_train))
# y_min = [0.5] * 7
# y_max = [0.5] * 7
# out = 0
# for i in range(len(dataset_train)//200):
#     y = dataset_train[i][1].cpu().detach().numpy()
#     for j in range(7):
#         if y[j] > 1 or y[j] < -1:
#             out += 1
#             #break
#         if y[j] > y_max[j]:
#             y_max[j] = y[j]
#         if y[j] < y_min[j]:
#             y_min[j] = y[j]
# print(y_min)
# print(y_max)
# print(out)

data_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
num_batches_train = len(dataset_train) // BATCH_SIZE + 1
num_batches_test = len(dataset_test) // BATCH_SIZE + 1

optim_pred = torch.optim.Adam(predator.parameters(), lr=LR)
optim_prey = torch.optim.Adam(prey.parameters(), lr=LR)

print(f"  Epoch  |     Predator loss     |       Prey loss     \n"
      f"         |   train   |    test   |   train   |    test \n"
      f"---------|-----------|-----------|-----------|-----------")


def train_batch(inputs, targets, train_losses, eval=False):
    num_batches = num_batches_test if eval else num_batches_train
    for i in range(2):
        idx = list(range(4 * i, 4 * i + 4)) + list(range(8, 33))
        inputs_i = inputs[:, idx]
        outputs = predator(inputs_i).squeeze(-1)
        loss = F.mse_loss(outputs, targets[:, i])
        if not eval:
            optim_pred.zero_grad()
            loss.backward()
            optim_pred.step()
        train_losses[0] += loss.item() / num_batches / 2

    for i in range(5):
        idx = list(range(8)) + list(range(8 + 5 * i, 13 + 5 * i))
        inputs_i = inputs[:, idx]
        outputs = prey(inputs_i).squeeze(-1)
        tar = targets[:, 2 + i]
        loss = F.mse_loss(outputs, tar)
        if not eval:
            optim_prey.zero_grad()
            loss.backward()
            optim_prey.step()
        train_losses[1] += loss.item() / num_batches / 5


for epoch in range(1, NUM_EPOCH + 1):
    predator.train()
    prey.train()
    train_losses = [0, 0]
    b = 0
    for inputs, targets in data_train:
        print(f'\rTraining {b}/{num_batches_train}', end='')
        inputs.to(device)
        targets.to(device)
        train_batch(inputs, targets, train_losses)
        b += 1

    if epoch % SAVE_EVERY == 0:
        predator.eval()
        prey.eval()
        test_losses = [0, 0]
        b = 0
        for inputs, targets in data_test:
            print(f'\rEvaluating {b}/{num_batches_test}', end='')
            inputs.to(device)
            targets.to(device)
            train_batch(inputs, targets, test_losses, eval=True)
            b += 1

        print(f"\r  {epoch:>5d}  |"
              f"  {train_losses[0]:>0.5f}  |  {test_losses[0]:>0.5f}  |"
              f"  {train_losses[1]:>0.5f}  |  {test_losses[1]:>0.5f}")
        torch.save(predator, f"models/predator{epoch}.pt")
        torch.save(prey, f"models/prey{epoch}.pt")
