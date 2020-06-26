# Model Averaging

Model averaging can be justified as being Bayesians about model uncertainty as we were Bayesians about parameter uncertainty. If we can not absolutely be sure that **a** model is **the** model (and generally we can not), then we should somehow take that uncertainty into account in our inferences. One way of doing this is by performing a weighted average of **all the considered models**, giving more weight to the models that seems to explain/predict the data better. To do this we can use both Information Criteria (IC) as well as the marginal likelihood.

## Pseudo Bayesian model averaging

Bayesian models can be weighted by their marginal likelihood, this is known as Bayesian Model Averaging. While this is theoretically appealing, it is problematic in practice. As we already discussed in the Bayes Factor section, the computation of the maginal likelihood can easily become a minefield. An alternative route is to use the values of WAIC or LOO to estimate weights for each model. We can do this by using the following formula:

$$w_i = \frac {e^{ - \frac{1}{2} dIC_i }} {\sum_j^M e^{ - \frac{1}{2} dIC_j }}$$

Where $dIC_i$ is the difference between the i-esim information criterion value and the lowest one. We are assuming that we are using the deviance scale. 

This approach is called pseudo Bayesian model averaging, or Akaike-like weighting and is an euristic way to compute the relative probability of each model (given a fixed set of models) from the information criteria values. See how the denominator is just a normalization term to ensure that the weights sum up to one.

## Pseudo Bayesian model averaging with Bayesian Bootstrapping

The above formula for computing weights is a very nice and simple approach, but with one major caveat it does not take into account the uncertainty in the computation of the IC. We could compute the standard error of the IC (assuming a Gaussian approximation) and modify the above formula accordingly. Or we can do something more robust, like using a [Bayesian Bootstrapping](http://www.sumsar.net/blog/2015/04/the-non-parametric-bootstrap-as-a-bayesian-model/) to estimate, and incorporate this uncertainty. This is the default approach used by ArviZ's `compare(.)` function.

## Stacking

Yet another option is known as [stacking](https://arxiv.org/abs/1704.02030) of predictive distributions. We want to combine several models in a meta-model in order to minimize the divergence between the meta-model and the _true_ generating model. When using a logarithmic scoring rule this is equivalently to compute:

$$\max_{n} \frac{1}{n} \sum_{i=1}^{n}log\sum_{k=1}^{K} w_k p(y_i|y_{-i}, M_k)$$

Where $n$ is the number of data points and $K$ the number of models. To enforce a solution we constrain $w$ to be $w_k \ge 0$ and  $\sum_{k=1}^{K} w_k = 1$. 

The quantity $p(y_i|y_{-i}, M_k)$ is the leave-one-out predictive distribution for the $M_k$ model. Computing it requires fitting each model $n$ times, each time leaving out one data point. Fortunately we can approximate the exact leave-one-out predictive distribution using LOO (or even WAIC), and that is what ArviZ does in practice when we call `compare(method="stacking")`.

## Weighted posterior predictive samples

Once we have computed the weights, using any of the above 3 methods (or in fact any other method we could think of), we can compute *weighted* posterior predictive samples. This is not something ArviZ computes but libraries like PyMC3 can do it for us:

#pp_weighted_samples = pm.sample_posterior_predictive_w(traces, weights=cmp.weight)
# add example here

## Other options

There are other ways to average models such as, for example, explicitly building a meta-model including all the models we have. We then perform parameter inference while *jumping* between the models. One problem with this approach is that *jumping* between models is difficult for most samplers. Besides averaging discrete models we can sometimes think of continuous versions of them. A toy example is to imagine that we have a coin and we want to estimate its degree of bias, which is a number between 0 and 1 -- being 0.5 equal chance of getting heads or tails. We could think of two separated models: one with a prior biased towards heads and one towards tails. We could fit both separate models and then average them using, for example, IC-derived weights. But we can also build a hierarchical model to estimate the prior from the data, and instead of contemplating two discrete models we will be computing a continuous model that includes these two discrete ones as particular cases. 
This begs the question: which approach is better? And as is generally the case in statistics the correct answer depends on our concrete problem. Do we have background information supporting two discrete models, or is our problem better represented with a continuous bigger model?