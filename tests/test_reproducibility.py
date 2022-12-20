from random import randrange

import numpy as np
import torch

from src.utils import seed_everything


def test_reproducibility():
    seed_everything()
    python_random_a = randrange(100)
    np_random_a = np.random.randint(100, size=1)[0]
    torch_random_a = torch.randint(100, size=(1, ))[0]

    seed_everything()
    python_random_b = randrange(100)
    np_random_b = np.random.randint(100, size=1)[0]
    torch_random_b = torch.randint(100, size=(1, ))[0]

    assert python_random_a == python_random_b
    assert np_random_a == np_random_b
    assert torch_random_a == torch_random_b
