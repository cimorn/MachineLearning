from easydict import EasyDict
import torch

global_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Env = EasyDict({
    "name": "Swimmer-v5",
    "state": {},
    "action": (-1.0, 1.0),
    "dim": {
        "state": 8,
        "action": 2,
        "reward": 1,
        "nextstate": 8,
        "Q": 1
    },
    "reward": (-float('inf'), float('inf')),
    "steps": 1000
})

Path = EasyDict({
    "data": {
        "raw": "output/data/raw",
        "final": "output/data/final"
    },
    "model": {
        "agent": "output/model/agent"
    },
    "results": {
        "figures": "output/results/figures",
        "table": "output/results/table"
    }
})

run = EasyDict({
    "generate": {
        "epochs": 50
    },
    "dim": {
        "state": Env.dim.state,
        "action": Env.dim.action,
        "Q": Env.dim.Q
    },
    "train": {
        "gamma": 0.99,
        "lr": 1e-5,
        "epochs": 500,
        "update": 5,
        "grad_steps": 1,
        "grad_lr": 1e-2,
        "batch": 100,
        "eval": 1
    },
    "eval": {
        "epochs": 100
    }
})
