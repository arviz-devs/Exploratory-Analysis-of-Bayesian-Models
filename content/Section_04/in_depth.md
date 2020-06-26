## WAIC in depth

WAIC, generally pronounced as W-A-I-C, even when something like *wæɪk* is much more easier ;-) is an estimator of the generalization loss which is defined as:


$$G_n = - \int q(x) \log p(x \mid X^n) dx$$

Where $X^n$ is a sample taking from the *true* distribution $q(x)$ and $p(x \mid X^n)$ is the predictive density. Notice that $G_n$ is a theoretical quantity. that assumes we known $q(x)$ and $p(x \mid X^n)$. Also notice that when doing posterior predictive checks we approximate $p(x \mid X^n)$, which in some sense provide a mathematical justification to posterior predictive checks and a connection between Information Criteria and posterior predictive checks. 


The generalization error is related to the concept of entropy that we can define as:

$$S = - \int q(x) \log q(x) dx$$

The entropy is....

If we take 

$$G_n - S = - \int q(x) \log p(x \mid X^n) dx + \int q(x) \log q(x) dx$$

Using the associative property and properties of logarithms

$$G_n - S = \int q(x) \log \frac{q(x)}{p(x \mid X^n)} dx$$

The term on the right hand side is known as Kullback-Leibler divergence from q(x) to p(x \mid X^n)


So we can think of $G_n$ as the theoretical 


AIC is founded on information theory. When a statistical model is used to represent the process that generated the data, the representation will almost never be exact; so some information will be lost by using the model to represent the process. AIC estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model.

In estimating the amount of information lost by a model, AIC deals with the trade-off between the goodness of fit of the model and the simplicity of the model. In other words, AIC deals with both the risk of overfitting and the risk of underfitting.

The Akaike information criterion is named after the statistician Hirotugu Akaike, who formulated it. It now forms the basis of a paradigm for the foundations of statistics; as well, it is widely used for statistical inference. 

## LOO in depth
