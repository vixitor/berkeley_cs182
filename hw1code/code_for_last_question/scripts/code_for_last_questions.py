from help_functions import generate_data, device, to_torch, to_numpy, backward_and_plot_grad
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from hw1code.code_for_last_question.scripts.help_functions import compute_svd_plot_features

width = 10
f = lambda x: np.sin(2 * np.pi * x)
model = nn.Sequential(nn.Linear(1, width),
                        nn.ReLU(),
                        nn.Linear(width, width),
                        nn.ReLU(),
                        nn.Linear(width, 1))
n_samples = 1000
n_features = 1
X = np.random.rand(n_samples, n_features)
X = np.sort(X, axis=0)  # Sort X for better visualization
y = generate_data(X, f, noise=0.1)

plt.scatter(X, y, s=5, label="data")
plt.legend()
plt.savefig("../img/train_data.png", dpi=300, bbox_inches="tight")
plt.close()

test_data = {}
test_data['X'] = np.concatenate((np.linspace(0, 1, 100).reshape(-1, 1), np.random.rand(100, 1)), axis=0)
test_data['X'] = np.sort(test_data['X'], axis=0)  # Sort for better visualization
print(f'test data X shape:{test_data["X"].shape}')
test_data['y'] = generate_data(test_data['X'], f, noise=0)

plt.scatter(test_data['X'], test_data['y'], s=5, label="test data")
plt.legend()
plt.savefig("../img/test_data.png", dpi=300, bbox_inches="tight")
plt.close()

loss = nn.MSELoss()
X_train = to_torch(X)
y_train = to_torch(y.reshape(-1, 1))

test_data['X'] = to_torch(test_data['X'])
test_data['y'] = to_torch(test_data['y'].reshape(-1, 1))

backward_and_plot_grad(X, model, "untrained")

n_steps = 10000
model = model.to(device)
optim = torch.optim.SGD(model.parameters(), lr=0.01)

for s in range(n_steps):
    subsample = np.random.choice(y.size, y.size // 5)
    step_loss = loss(y_train[subsample], model(X_train[subsample, :]))
    model.zero_grad()
    step_loss.backward()
    optim.step()
    if (s + 1) % 1000 == 0 or s == 0:
        print(f"Step {s + 1}, train loss: {step_loss.item()}")
        with torch.no_grad():
            test_loss = loss(test_data['y'].reshape(-1, 1), model(test_data['X']))
            print(f"Test loss: {test_loss.item()}")

backward_and_plot_grad(X, model, "trained")
compute_svd_plot_features(X, model)


