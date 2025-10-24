import math
import torch
import numpy as np


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, out_dir=None, is_save_task_pool=True, **kwargs
):
    task_cls = FourierSeriesV3
    return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)


class FourierSeriesV3(Task):
    """
        Constant SNR version: 
                SNR = d var(w) / var(epsilon)
    """
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_frequency=10,
        min_frequency=1,
        L=5,
        standardize=False,
        intercept=False,
        tail_idx=None,
        sigma2_y_adj=1,
        weight_prior=1.,
        noise_type='gaussian',
        dof_t=3.0
    ):
        super(FourierSeriesV3, self).__init__(n_dims, batch_size, pool_dict, seeds)
        assert n_dims == 1

        self.scale = scale
        self.L = L
        self.standardize = standardize
        self.noise_type = noise_type
        self.batch_size = batch_size

        if self.noise_type == 'student_t':
            self.dof_t = dof_t
            self.sampler = torch.distributions.studentT.StudentT(torch.tensor([dof_t]))

        # Assign max_frequency for each sample in the batch
        if tail_idx is None:
            self.max_frequencies = np.full(self.batch_size, max_frequency)
        else:
            frs = 1 / np.arange(1, 1 + max_frequency)**(float(tail_idx))
            frs = frs / frs.sum()
            self.max_frequencies = np.random.choice(
                np.arange(1, 1 + max_frequency), 
                size=self.batch_size, 
                p=frs
            )

        self.epsilon = np.sqrt(0.25 / 8. * sigma2_y_adj)

        # Generate coefficients per sample
        self.a_coefs = torch.zeros(self.batch_size, max_frequency + 1)
        self.b_coefs = torch.zeros(self.batch_size, max_frequency)

        for i, mf in enumerate(self.max_frequencies):
            coefs = torch.randn(1, 2 * mf + 1) * np.sqrt(weight_prior) / np.sqrt(2 * mf + 1)
            a_coefs = coefs[:, :mf + 1]
            b_coefs = coefs[:, mf + 1:]

            # Apply frequency masks
            a_freq_mask = torch.zeros(1, mf + 1)
            a_freq_mask[:, 0] = 1
            a_freq_mask[:, min_frequency:] = 1
            b_freq_mask = torch.zeros(1, mf)
            b_freq_mask[:, min_frequency - 1:] = 1

            self.a_coefs[i, :mf + 1] = (a_coefs * a_freq_mask).squeeze()
            self.b_coefs[i, :mf] = (b_coefs * b_freq_mask).squeeze()

    def evaluate(self, xs_b):
        device = xs_b.device
        batch_size = xs_b.size(0)
        sequence_length = xs_b.size(1)

        # Prepare the output tensor
        out = torch.zeros(batch_size, sequence_length).to(device)

        # Compute for each batch
        for i in range(batch_size):
            mf = self.max_frequencies[i]
            a_coefs = self.a_coefs[i, :mf + 1].to(device)
            b_coefs = self.b_coefs[i, :mf].to(device)

            cosine_terms = (
                a_coefs.unsqueeze(0)
                * torch.cos(
                    (math.pi / self.L)
                    * torch.arange(mf + 1).unsqueeze(0).to(device)
                    * xs_b[i].view(-1, 1)
                )
            ).sum(axis=-1)

            sine_terms = (
                b_coefs.unsqueeze(0)
                * torch.sin(
                    (math.pi / self.L)
                    * torch.arange(1, mf + 1).unsqueeze(0).to(device)
                    * xs_b[i].view(-1, 1)
                )
            ).sum(axis=-1)

            sample_out = self.scale * (cosine_terms + sine_terms)
            out[i] = sample_out

        # Add noise if required
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(out) * self.epsilon
        elif self.noise_type == 'student_t':
            if self.dof_t > 2:
                var_t = self.dof_t / (self.dof_t - 2)
                noise_adj = np.sqrt(1 / var_t)
            else:
                noise_adj = np.sqrt(1 / 3)
            noise = self.sampler.sample(sample_shape=out.shape) * self.epsilon * noise_adj
            noise = noise.squeeze()
        else:
            noise = 0

        return out + noise

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error



