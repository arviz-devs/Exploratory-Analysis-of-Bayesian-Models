# Diagnosing the samples from MCMC

The Achilles heel of computing posterior distributions is, most often than not, the computation of the denominator of Bayes's theorem. Markov Chain Monte Carlo Methods (MCMC), such as Metropolis, Hamiltonian Monte Carlo (and its variants like NUTS) are clever mathematical and computational devices that let's us circumvent this problem. The main idea is based on sampling values of the parameters of interest from an easy to sample distribution and then applying a rule, known as metropolis acceptance criterion, to decide if you accept or not that proposal. This rule is necessary as it provides the proper way of correcting those samples to get samples from the true distribution, _i.e_ the posterior distribution in a Bayesian Analysis. The better performance of NUTS over Metropolis can be explained by the fact that the former uses a clever way to propose values. 


While MCMC methods can be successfully used to solve a huge variety of Bayesian models, this have some trade-offs. Most notably, MCMC methods can be slow for some problems, such as _Big Data_ problems and what is most important for our current discussion finite MCMC chains are not guaranteed to converge to the true parameter value. Thus, is important to check whether we have a valid sample, otherwise any analysis from it will be totally flawed. This section is focused on diagnosing sample obtained from Markov Chain Monte Carlo Methods.

There are several tests we can perform, some are visual and some are quantitative. These tests are designed to spot problems with our samples, but they are unable to prove we have the correct distribution; they can only provide evidence that the sample seems reasonable.


If we find problems with the samples, the are many solutions to try:
* Increase the number of samples.
* Remove a number of samples from the beginning of the trace. This is know as burn-in. Software such as PyMC3 and PyStan use a tuning/warm-up phase that helps reduce the need for burn-in. 
* Modify sampler parameters, such as increasing the length of the tuning phase, or increase the `target_accept` parameter for the NUTS sampler. Under certain circumstances, PyMC3 will provide suggestions on what to change.
* Transform the data, eg center of standardize it.
* Re-parametrize the model, that is, express the model in a different but equivalent way, we will see examples of this in the next sections.