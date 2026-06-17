### <ins>Naive Bayes classifier

Naive Bayes classifier is a probabilistic supervised machine learning model that is used to classify samples into classes.

Given a training set the model computes the probability of each class as well as the probability for each feature to appear in any of the classes.

**$P(C_1)$** is the probability of class 1.

**$P(C_0)$** is the probability of class 0.

$P(X | C)$ is the probability of the features given the class.


The assumption that the algorithm does and simplifies the model is that it assumes that all features are independent which means that $$P(X|C)$$ can be decomposed down to the multiplication of all Conditional probabilities of the features given the class times the probability of observing the class.

$$P(X|C) = P(x_1|C)P(x_2|C)...P(x_N|C)P(C)  $$

After computing the probability of each class the most probable class is considered as the models final prediction.

***Note:***
When the explanatory variable are categorical the probabilities are computed by counting. However when the explanatory variables are continuous the model assumes that for each feature:
$$
P(x_i \mid C) \sim \mathcal{N}(\mu_{C,i}, \sigma_{C,i}^2)
$$

