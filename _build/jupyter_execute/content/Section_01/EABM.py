# Exploratory analysis of Bayesian models

While conceptually simple, Bayesian methods can be mathematically and numerically
challenging. Probabilistic programming languages (PPLs) implement functions to easily
build Bayesian models together with efficient automatic inference methods. This helps
separate the model building from the inference, allowing practitioners to focus on their
specific problems and leaving PPLs to handle the computational details for them ([Bessiere, Mazer, Ahuactzin, & Mekhnacha, 2013](https://www.crcpress.com/Bayesian-Programming/Bessiere-Mazer-Ahuactzin-Mekhnacha/p/book/9781439880326); [Daniel Roy, 2015](http://probabilistic-programming.org/wiki/Home); [Ghahramani, 2015](https://doi.org/10.1038/nature14541)). The inference process generates a posterior distribution - which has a central role in Bayesian statistics - together with other distributions like the posterior predictive distribution and the prior predictive distribution. The correct visualization, analysis, and interpretation of these distributions is key to properly answer the questions that motivate the inference process.

When working with Bayesian models there are a series of related tasks that need to be
addressed besides inference itself:

* Diagnoses of the quality of the inference
* Model criticism, including evaluations of both model assumptions and model predictions
* Comparison of models, including model selection or model averaging
* Preparation of the results for a particular audience

Successfully performing such tasks are central to the iterative and interactive modeling
process. These tasks require both numerical and visual summaries to help statisticians
or practitioners analyze visual summaries. In the words of [Persi Diaconis](https://statistics.stanford.edu/research/theories-data-analysis-magical-thinking-through-classical-statistics).

> Exploratory data analysis (EDA) seeks to reveal structure, or simple descriptions in data. We look at numbers or graphs and try to find patterns. We pursue leads suggested by background information, imagination, patterns perceived, and experience with other data
analyses.

**Exploratory analysis of Bayesian models** take many of the ideas commonly associated with EDA and apply them to the analysis of Bayesian models and their results.

In the following sections we will discuss how to use ArviZ to perform Exploratory analysis of Bayesian models, together with a discussion of good practices for Bayesian modeling. As visualization plays a central role in Exploratory analysis of Bayesian models we begin our discussion with some general topics around visualization.