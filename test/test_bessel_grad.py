import torch
import numpy as np
from scipy.special import iv, ive

from loss import VonMisesFisherLoss

input_dim = 400
vmf_fs = VonMisesFisherLoss(input_dim, reduction='none', use_finite_sums=True, n_bessel_iters=100)
vmf_lb = VonMisesFisherLoss(input_dim, reduction='none', use_finite_sums=False)
mu = torch.Tensor(np.random.uniform(-.5, .5, size=input_dim))
mu /= mu.norm()

xs = [
np.random.uniform(-.5, .5, size=input_dim),
np.random.uniform(-.5, .5, size=input_dim),
        ]

#def get_loss(x, mu, loss_fn):
def get_loss(dim, x, loss_fn):
    x = torch.autograd.Variable(torch.Tensor([x]), requires_grad=True)
    #x = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
    #loss = loss_fn(x, mu).sum()
    loss = loss_fn(dim, x) # now dim and x
    loss.backward()
    return loss, x.grad

def scipy_grad(v, z):
    grad = iv(v/2, z)/iv(v/2-1, z)
    return grad

def norm(x):
    return np.sqrt((x ** 2).sum())

def compare(x, mu, f1, f2):
    x = np.array(x)
    #loss1, x1grad = get_loss(x, mu, f1)
    #loss2, x2grad = get_loss(x, mu, f2)
    dim = torch.Tensor([input_dim])
    loss1, x1grad = get_loss(dim, norm(x), f1)
    loss2, x2grad = get_loss(dim, norm(x), f2)
    x3grad = scipy_grad(input_dim, norm(x))
    #print(f"mu:\t{mu.numpy()}")
    #print(f"x:\t{x}")
    print(f"loss1:\t{loss1}")
    print(f"loss2:\t{loss2}")
    print(f"dx1:\t{x1grad.sum() - x3grad}")
    print(f"dx2:\t{x2grad.sum() - x3grad}")
    # print(f"dx3:\t{x3grad}")
    print()

for x in xs:
    compare(x, mu, vmf_fs, vmf_lb)
