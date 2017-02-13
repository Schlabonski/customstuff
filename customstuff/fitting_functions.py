#!/usr/bin/python3

import numpy as np

def GaussC(x, mu, sigma, A, c):
  """Returns the probability of x when chosen from a gaussian ditribution with
  constant offset.

  :x: float
  :mu: float, mean value of distro
  :sigma: float, variance of distro
  :A: float, for non-normal distros
  :c: float, constant offset
  :returns: float, probability of x

  """
  P = A/(sigma*np.sqrt(np.pi))* np.exp(-(x - mu)**2/(2*sigma**2)) + c
  return P

def GaussCL(x, mu, sigma, A, b, c):
  """Probability of x when chosen from a given gaussian distribution
  with linear offset.

  :x: float
  :mu: float, mean value of distro
  :sigma: float, variance of distro
  :A: float, for non-normal distros
  :b: float, slope of underlying linear function
  :c: float, constant offset
  :returns: float, probability of x

  """
  P = A/(sigma*np.sqrt(np.pi))* np.exp(-(x - mu)**2/(2*sigma**2)) + b*x + c
  return P

def LorentzDistribution(x, x0, gamma, A):
  """Probability of x, when chosen from Lorentz Distribution 

  :x: float
  :x0: float, median of distro
  :gamma: float, scale parameter of distro
  :A: float, scaling for non-normalized distro
  :returns: float, probability of x

  """
  P = A/(np.pi*gamma*(1+((x-x0)/gamma)**2))
  return P

def ExponentialC(A, tau, c, t_0):
    """Exponential curve with constant offsets in x and y direction

    :A: prefactor to the exponential
    :tau: inverse decay rate
    :c: constant y offset
    :t_0: constant x offset

    """
    return A*np.exp((t-t_0)/tau) + c
