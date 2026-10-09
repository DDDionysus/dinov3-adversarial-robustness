import torch


def generate_random_delta(x, epsilon, seed):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    delta = torch.empty_like(x).uniform_(-epsilon, epsilon, generator=generator)

    return delta
