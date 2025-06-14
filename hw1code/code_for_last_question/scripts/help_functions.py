from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device=device)
    elif isinstance(x, torch.Tensor):
        return x.float().to(device=device)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Input must be a torch tensor or a numpy array.")

def generate_data(X, f, noise=0.1):
    return f(X[:, 0]) + noise * np.random.randn(X.shape[0])

def backward_and_plot_grad(X, model, name="untrained"):
    model.to(device)
    grad_collect = {}
    vis_collect = []
    for x in X:
        y = model(to_torch(x.reshape(1, -1)))
        model.zero_grad()
        model.retain_gradients = True
        y.backward()
        for n, p in model.named_parameters():
            if n not in grad_collect:
                grad_collect[n] = []
            grad_collect[n].append(to_numpy(p.grad.data).ravel())
            if n not in vis_collect:
                vis_collect.append(n)
    for n in vis_collect:
        plt.plot(X, grad_collect[n], label=n)
        plt.xlabel("Input")
        plt.ylabel("Gradient")
        plt.title(f"Gradient of {n} over input")
        plt.legend()
        plt.savefig(f"../img/{name}_grad_{n}.png")
        plt.close()





