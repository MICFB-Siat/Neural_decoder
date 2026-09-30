import torch

from sgm.modules.diffusionmodules.discretizer import Discretization


class Img2ImgDiscretizationWrapper:






    def __init__(self, discretization: Discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):

        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:






    def __init__(
        self, discretization: Discretization, strength: float = 0.0, original_steps=None
    ):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):

        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        print("prune index:", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas
