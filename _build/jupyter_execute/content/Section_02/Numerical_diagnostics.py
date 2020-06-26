import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc3 as pm
import arviz as az

# Numerical Diagnostics

We will discuss 3 numerical diagnostics available in ArviZ, those are:

* Effective Sampler Size
* $\hat R$ (R hat)
* mcse error

To help us understand these diagnostics we are going to create two _synthetic posteriors_. The first one is a sample from a uniform distribution. We generate it using SciPy and we call it `good_chains`. This is an example of a "good" sample because we are generating independent and identically distributed (iid) samples and ideally this is what we want to approximate the posterior. The second one is called `bad_chains`, and it will represent a poor sample from the posterior. `bad_chains` is a poor _sample_ for two reasons:

* Values are not independent. On the contrary they are highly correlated, meaning that given any number at any position in the sequence we can compute the exact sequence of number both before and after the given number. Highly correlation is the opposite of independence.
* Values are not identically distributed, as you will see we are creating and array of 2 columns, the first one with numbers from 0 to 0.5 and the second one from 0.5 to 1.

good_chains = stats.uniform.rvs(0, 1,size=(2,500))
bad_chains = np.linspace(0, 1, 1000).reshape(2, -1)

## Effective Sample Size (ESS)

When using sampling methods like MCMC is common to wonder if a particular sample is large enough to confidently compute what we want, like for example a parameter mean. Answering in terms of the number of samples is generally not a good idea as samples from MCMC methods will be autocorrelated and autocorrelation decrease the actual amount of information contained in a sample. Instead, a better idea is to estimate the **effective Sample Size**, this is the number of samples we would have if our sample were actually iid. 

Using ArviZ we can compute it using `az.ess(⋅)`

az.ess(good_chains), az.ess(bad_chains)

This is telling us that even when in both cases we have 1000 samples, `bad_chains` is somewhat equivalent to a iid sample of size $\approx 2$. While `good_chains` is $\approx 1000$. If you resample `good_chains` you will see that the effective sample size you get will be different for each sample. This is expected as the samples will not be exactly the same, they are after all samples. Nevertheless, on average, the value of effective sample size will be lower than the $N$ number of samples. Notice, however, that ESS could be in fact larger! When using the NUTS sampler value pf $ESS > N$ can happen for parameters which posterior distribution close to Gaussian and which are almost independent of other parameters.

> As a general rule of thumb we recommend an `ess` greater than 50 per chain, otherwise the estimation of the `ess` itself and the estimation of $\hat R$ are most likely unreliable.

Because MCMC methods can have difficulties with mixing, it is important to use between-chain information in computing the ESS. This is one reason to routinary run more than one chain when fitting a Bayesian model using MCMC methods.

We can also compute the effective sample size using `az.summary(⋅)`

az.summary(good_chains)

As you can see `az.summary(⋅)` provides 4 values for `ESS`, mean, sd, bulk and tail. Even more if you check the arguments `method` of the `az.ess(⋅)`  functions you will see the following options

- "bulk"
- "tail"
- "quantile"
- "mean"
- "sd"
- "median"
- "mad"
- "z_scale"
- "folded"
- "identity"


Why in hell ArviZ offers so many options? Just to make you life miserable, not just kidding, these estimates correspond to the effective sample size for different "parts" of your distribution. The reason we need this is that the mixing of Markov chains is not uniform across the parameter space. Thus the ESS estimate for the center of the distribution (the ess-bulk) could be different from that from the tails (ess-tail)

## Effective Sample Size in depth


The basic ess diagnostic is computed by:

$$\hat{N}_{eff} = \frac{MN}{\hat{\tau}}$$

where $M$ is the number of chains, $N$ the number of draws per chain and $\hat t$ is a measure of the autocorrelation in the samples. More precisely $\hat t$ is defined as follows:

$$\hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_{t'}$$

where $\hat{\rho}_t$ is the estimated autocorrelation at lag $t$, and $K$ is the largest integer for which $\hat{P}_{K} = \hat{\rho}_{2K} + \hat{\rho}_{2K+1}$ is still positive. The reason to compute this truncated sum, we are summing over $K$ terms instead of summing over all available terms is that for large values of $t$ the sample correlation becames too noisy to be useful, so we simply discard those terms in order to get more robust estimate.

## $\hat R$ (aka R hat, or Gelman-Rubin statistics)


Under very general conditions MCMC methods have theoretical guarantees that you will get the right answer irrespective of the starting point. Unfortunately, we only have guarantee for infinite samples. One way to get a useful estimate of convergence for finite samples is to run more than one chain, starting from very different points and then checking if the resulting chains _look similar_ to each other. $\hat R$ is a formalization of this idea and it works by comparing the the _in chain_ variance to the _between chain_ variance. Ideally we should get a valuer of 1.

Conceptually $\hat R$ can be interpreted as the overestimation of variance due to MCMC finite sampling. If you continue sampling infinitely you should get a reduction of the variance of your estimation by a $\hat R$ factor.

From a practical point of view $\hat R \lessapprox 1.01$ are considered safe

Using ArviZ we can compute it using `az.summary(⋅)`, as we already saw in the previous section or using  `az.rhat(⋅)`

az.rhat(good_chains), az.rhat(bad_chains)

## $\hat R$ in depth


The value of $\hat R$ is computed using the between-chain variance $B$ and within-chain variance $W$, and then assessing if they are different enough to worry about convergence. For $M$ chains, each of length $N$, we compute for each scalar parameter $\theta$:

\begin{split}B &= \frac{N}{M-1} \sum_{m=1}^M (\bar{\theta}_{.m} - \bar{\theta}_{..})^2 \\
W &= \frac{1}{M} \sum_{m=1}^M \left[ \frac{1}{N-1} \sum_{n=1}^n (\theta_{nm} - \bar{\theta}_{.m})^2 \right]\end{split}

where:

$\bar{\theta}_{.m} = \frac{1}{N} \sum_{n=1}^N \theta_{nm}$

$\bar{\theta}_{..} = \frac{1}{M} \sum_{m=1}^M \bar{\theta}_{.m}$

Using these values, an estimate of the marginal posterior variance of $\theta$ can be calculated:

$$\hat{\text{Var}}(\theta | y) = \frac{N-1}{N} W + \frac{1}{N} B$$

Assuming $\theta$ was initialized using overdispersed starting points in each chain, this quantity will overestimate the true marginal posterior variance. At the same time, $W$ will tend to underestimate the within-chain variance early in the sampling run, because the individual
chains have not had the time to explore the entire target distribution. However, in the limit as $n \to \infty$, both quantities will converge to the true variance of $\theta$. 

Finally, we compute the $\hat R$ statistic as:

$$\hat{R} = \sqrt{\frac{\hat{\text{Var}}(\theta | y)}{W}}$$

For an ergodic chain $\hat{R}$ will converge to 1 $n \to \infty$. In practice $\hat{R}$ is computed by splitting the chain in half so $M$ is two times the number of chains. This is a simply trick to ensure that the first and last parts of a chain are indeed similar as expected from a converged chain.

# Monte Carlo Standard Error  

When using MCMC methods we introduce an additional layer of uncertainty, due to the finite sampling, we call this Monte Carlo Standard Error (mcse). The mcse takes into account that the samples are not truly independent of each other. If we want to report the value of an estimated parameter to the second decimal we need to be sure the mcse error is below the second decimal otherwise we will be, wrongly, reporting a higher precision than we really have. We should check the mcse error once we are sure $\hat R$ is low enough and ESS is high enough, otherwise mcse error is of no use.  

Using ArviZ we can compute it using `az.mcse(⋅)`

az.mcse(good_chains), az.mcse(bad_chains)

## mcse in depth

To compute the mcse the chain is divided into $n$ batches, for each batch we computes its mean and then we compute the standard deviation of those means divided by the square root of the $n$ batches.

$$\text{mcse} = \frac{\sigma(x)}{\sqrt{n}}$$

## Summary

The ESS statistics answer the question is the chain large enough? while the $\hat R$ diagnostics answers the question _did the chains mix well?_. Finally the mcse error estimates the amount of error introduced by sampling and thus the level of precision of our estimates.