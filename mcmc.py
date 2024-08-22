"""Markov chain Monte Carlo methods"""

import torch
import numpy as np
from tqdm import tqdm

# z-normalization functions
def z_normalize(params, means, std_devs):
    return (params - means) / std_devs

# inverse z-normalization
def inverse_z_normalize(z_params, means,  std_devs):
    return z_params * std_devs + means

def log_normal_density(x, **kwargs):
    """
    Computes the log probability density of a normal distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.

    Returns:
        Log probability density at the given input tensor x.
    """
    for key, value in kwargs.items():
        kwargs[key] = torch.tensor([value])
    mu, sigma = kwargs['mu'], kwargs['sigma']

    return (
        -0.5*np.log(2*np.pi) -                      # constant term
        torch.log(sigma) -                          # logarithm of the determinant of the covariance matrix
        0.5*(((x-mu)/sigma)**2)                     # exponential term
    )

def log_gamma_density(x, **kwargs):
    """
    Computes the log probability density of a Gamma distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - alpha: Shape parameter of the Gamma distribution.
    - beta: Rate parameter (1/scale) of the Gamma distribution.

    Returns:
        Log probability density at the given input tensor x.
    """
    for key, value in kwargs.items():
        kwargs[key] = torch.tensor([value])
    alpha, beta = kwargs['alpha'], kwargs['beta']   # shape, scale
    return alpha * torch.log(beta) + (alpha - 1) * torch.log(x) - beta * x - torch.lgamma(alpha)

def log_truncated_normal(x, **kwargs):
    """
    Computes the log probability density of a truncated normal distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.
    - a: Lower truncation bound.
    - b: Upper truncation bound.

    Returns:
        Log probability density at the given input tensor x.
    """
    for key, value in kwargs.items():
        kwargs[key] = torch.tensor([value])
    mu, sigma, a, b = kwargs['mu'], kwargs['sigma'], kwargs['a'], kwargs['b']
    # Standardize
    standard_x = (x - mu) / sigma
    standard_a = (a - mu) / sigma
    standard_b = (b - mu) / sigma
    
    # PDF of the standard normal distribution
    pdf = torch.exp(-0.5 * standard_x**2) / np.sqrt(2 * torch.pi)
    
    # CDF of the standard normal distribution using the error function
    cdf_upper = 0.5 * (1 + torch.erf(standard_b / np.sqrt(2)))
    cdf_lower = 0.5 * (1 + torch.erf(standard_a / np.sqrt(2)))
    
    normalization = cdf_upper - cdf_lower
    
    return torch.log(pdf) - torch.log(normalization)

# mapper dictionary - maps to appropriate log probability density function for given distribution `type`
function_map = {
    'normal': log_normal_density,
    'gamma': log_gamma_density,
    'truncated_normal': log_truncated_normal
}

def acceptance_rate(P_chain, window_size=100):
    return np.convolve(P_chain, np.ones(window_size)/window_size, mode='valid')

def effective_n(x):
    """
    Compute the effective sample size of a trace.

    Takes the trace and computes the effective sample size
    according to its detrended autocorrelation.

    Parameters
    ----------
    x : list or array of chain samples

    Returns
    -------
    effective_n : int
        effective sample size
    """
    # detrend trace using mean to be consistent with statistics
    # definition of autocorrelation
    x = np.asarray(x)
    x = (x - x.mean())
    # compute autocorrelation (note: only need second half since
    # they are symmetric)
    rho = np.correlate(x, x, mode='full')
    rho = rho[len(rho)//2:]
    # normalize the autocorrelation values
    #  note: rho[0] is the variance * n_samples, so this is consistent
    #  with the statistics definition of autocorrelation on wikipedia
    # (dividing by n_samples gives you the expected value).
    rho_norm = rho / rho[0]
    # Iterate until sum of consecutive estimates of autocorrelation is
    # negative to avoid issues with the sum being -0.5, which returns an
    # effective_n of infinity
    negative_autocorr = False
    t = 1
    n = len(x)
    while not negative_autocorr and (t < n):
        if not t % 2:
            negative_autocorr = sum(rho_norm[t-1:t+1]) < 0
        t += 1
    return int(n / (1 + 2*rho_norm[1:t].sum()))

# mass balance posterior class
class mbPosterior:
    def __init__(self, mb_obs, sigma_obs, priors, mb_func):
        self.mb_obs = mb_obs
        self.sigma_obs = sigma_obs
        self.prior_params = priors
        self.mb_func = mb_func

        # get mean and std for each parameter type
        self.means = torch.tensor([params['mu'] if 'mu' in params else 0 for params in priors.values()])
        self.stds = torch.tensor([params['sigma'] if 'sigma' in params else 1 for params in priors.values()])

    def log_prior(self, m):
        log_prior = []
        for i, (key, params) in enumerate(self.prior_params.items()):
            params_copy = params.copy()
            prior_type = params_copy.pop('type')
            function_to_call = function_map[prior_type]
            log_prior.append(function_to_call(m[i], **params_copy))
        log_prior = torch.stack(log_prior).sum()
        return log_prior

    def log_likelihood(self, m):
        # Denormalize the parameters before calculating likelihood
        mb_pred = self.mb_func([*m])
        return log_normal_density(self.mb_obs, **{'mu': mb_pred, 'sigma': self.sigma_obs})
    
    def log_posterior(self,m):
        return self.log_prior(m) + self.log_likelihood(m)
    
# Metropolis-Hastings Markoc chain Monte Carlo class
class Metropolis:
    def __init__(self, means, stds):
        # Initialize chains
        self.steps = []
        self.P_chain = []
        self.m_chain = []
        self.m_primes = []
        self.means = means
        self.stds = stds

    def sample(self, m_0, log_posterior, h=0.1, n_samples=1000, burnin=0, thin_factor=1, progress_bar=False):
        # Compute initial unscaled log-posterior
        P_0 = log_posterior(inverse_z_normalize(m_0, self.means, self.stds))

        n = len(m_0)

        # Draw samples
        iterable = range(n_samples)
        if progress_bar:
            iterable = tqdm(iterable)

        for i in iterable:
            # Propose new value according to
            # proposal distribution Q(m) = N(m_0,h)
            step = torch.randn(n)*h
            m_prime = m_0 + step

            # record step
            self.steps.append(step)

            # Compute new unscaled log-posterior
            P_1 = log_posterior(inverse_z_normalize(m_prime, self.means, self.stds))

            # Compute logarithm of probability ratio
            log_ratio = P_1 - P_0

            # Convert to non-log space
            ratio = torch.exp(log_ratio)

            # If proposed value is more probable than current value, accept.
            # If not, then accept proportional to the probability ratios
            if ratio>torch.rand(1):
                m_0 = m_prime
                P_0 = P_1

            # Only append to the chain if we're past burn-in.
            if i>burnin:
                # Only append every j-th sample to the chain
                if i%thin_factor==0:
                    self.P_chain.append(P_0)
                    self.m_chain.append(m_0)
                    self.m_primes.append(m_prime)

        return torch.vstack(self.m_primes), torch.vstack(self.steps), torch.tensor(self.P_chain), torch.vstack(self.m_chain)