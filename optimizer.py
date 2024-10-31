from typing import Callable, Iterable, Tuple
from numpy import zeros
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        print("wait")
        # self.state = []
        # for gid, group in enumerate(self.param_groups):
        #     self.state.append([(torch.zeros_like(p), torch.zeros_like(p)) for p in group["params"]])

        self.t = 0

    def step(self, closure: Callable = None):
        self.t += 1
        loss = None
        if closure is not None:
            loss = closure()

        for gid, group in enumerate(self.param_groups):
            for pid, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()


                # State should be stored in this dictionary
                state = self.state[p]
                if not state:
                    state['m'] = 0.0
                    state['v'] = 0.0

                m, v = state['m'], state['v']


                b1, b2 = group["betas"]
                m = b1*m + (1.0 - b1)*grad
                v = b2*v + (1.0 - b2)*torch.square(grad)

                # self.state[gid][pid] = (m, v)
                state['m'], state['v'] = m, v

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                decay = group["weight_decay"]
                # alt = alpha * math.sqrt(1 - math.pow(b2, self.t)) / (1 - math.pow(b1, self.t))

                # Bias correction
                # mcap = m / (1 - math.pow(b1, self.t))
                # vcap = v / (1 - math.pow(b2, self.t))
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                alt = alpha * math.sqrt(1 - math.pow(b2, self.t)) / (1 - math.pow(b1, self.t))

                # Update parameters
                # p.data = p.data - (alt * m / (torch.sqrt(v) + 1e-8) + decay * p.data)
                p.data = p.data - alt * m / (torch.sqrt(v) + group['eps']) # - alpha*decay*p.data
                p.data = p.data - alpha*decay*p.data

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss
