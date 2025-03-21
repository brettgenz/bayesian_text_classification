import pymc as pm

with pm.Model() as logistic_model:
    # establish priors for coefficients
    coefficients = pm.Normal('coefficient', mu=0, sigma=1, shape=X.shape[1])
    intercept = pm.Normal('intercept', mu=0, sigma=1)

    # linear combinatino (logits)
    logits = intercept + pm.math.dot(X, coefficients)

    # likelihood (Bernoulli)
    likelihood = pm.Bernoulli('likelihood', logit_p=logits, observed=y)

    # inference (sampling posterior)
    trace = pm.sample(1000, tune=1000, target_accept=0.9)

    