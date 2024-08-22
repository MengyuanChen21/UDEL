import time

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def obtain_extreme_max_alpha(alpha):
    bz, n_cls = alpha.shape
    _, max_indices = torch.max(alpha, dim=1)
    max_alpha = torch.sum(alpha, dim=1) - n_cls

    extreme_max_alpha = torch.ones_like(alpha, device='cuda')
    for i in range(bz):
        extreme_max_alpha[i, max_indices[i]] = max_alpha[i]

    return extreme_max_alpha


def calculate_wasserstain_distance(p_alpha, q_alpha, blur, scaling, n_sample):
    # start = time.time()
    bz, n_cls = p_alpha.shape
    wass_list = []
    dir_p = torch.distributions.dirichlet.Dirichlet(p_alpha)
    dir_q = torch.distributions.dirichlet.Dirichlet(q_alpha)
    samples_p = dir_p.sample(sample_shape=[n_sample]).transpose(0, 1)
    samples_q = dir_q.sample(sample_shape=[n_sample]).transpose(0, 1)

    # Compute the Wasserstein-2 distance between our samples,
    # with a small blur radius and a conservative value of the
    # scaling "decay" coefficient (.8 is pretty close to 1):
    Loss = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling)

    for i in range(bz):
        wass = Loss(samples_p[i], samples_q[i])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wass_list.append(wass)

    # end = time.time()
    wass_list_tensor = torch.tensor(wass_list)
    # print(f'Average Wasserstein distance:{wass_list_tensor.mean():.3f}, computed in {end - start:.3f}s.')

    return wass_list_tensor


def obtain_extreme_target_alpha(alpha, target):
    bz, n_cls = alpha.shape
    evidence_per_target_class = (torch.sum(alpha, dim=1) - n_cls) / (target.sum(dim=1) + 1e-3)
    extreme_target_alpha = torch.ones_like(alpha, device='cuda')
    extreme_target_alpha += target * evidence_per_target_class[:, None]
    return extreme_target_alpha


def obtain_aleatoric_uct(alpha, target, blur=0.05, scaling=0.8, n_sample=100):
    return calculate_wasserstain_distance(
        alpha,
        obtain_extreme_target_alpha(alpha, target),
        blur=blur,
        scaling=scaling,
        n_sample=n_sample
    )
