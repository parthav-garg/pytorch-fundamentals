import torch
import torch.nn as nn


def discretize(A, B, delta):
    delta = delta.unsqueeze(-1)
    A = A.unsqueeze(0).unsqueeze(0)
    B = B.unsqueeze(0).unsqueeze(0)
    del_a = delta * A
    A_bar = torch.exp(del_a)
    del_b = delta * B
    B_bar = (torch.special.expm1(del_a)  * del_b )/ (del_a + 1e-8) 
    return A_bar, B_bar


def mamba_discretize(A, B, delta):
    delta = delta.unsqueeze(-1)
    A = A.unsqueeze(0).unsqueeze(0)
    B = B.unsqueeze(2)
    del_a = delta * A
    A_bar = torch.exp(del_a)
    del_b = delta * B
    B_bar = (torch.special.expm1(del_a)  * del_b )/ (del_a + 1e-8) 
    return A_bar, B_bar


class Mamba(nn.Module):
    def __init__(self, A, B, delta):
        super(Mamba, self).__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.delta = nn.Parameter(delta)