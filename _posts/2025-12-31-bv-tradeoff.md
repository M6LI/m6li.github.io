---
title: The bias-variance trade-off isn't always accurate
subtitle: We explain the bias-variance trade-off, it's usefulness and why it can sometimes fail.
layout: default
date: 2025-12-31
keywords: machine-learning
published: true
---

In this post, I discuss one of the most fundamental concepts in supervised machine learning and its limitations.

The bias–variance trade-off is a fundamental concept popularised in the late $1900s$ that is used to describe how the complexity of a model affects the prediction error. It is taught in most foundational machine learning classes. Let's take a look at what it means, why it is significant, and—most intriguingly—why it isn't always true.

### The mean-square error decomposition

Let's suppose we have an input $x \in \mathbb{R}$, and that the true data-generating process is given by
\[
Y = f(x) + \epsilon, \qquad \mathbb{E}(\epsilon) = 0, \quad \mathbb{E}(\epsilon^2) = \sigma^2,
\]
where $\epsilon$ represents noise. Suppose we also have a predictive model $\widehat{Y} = \widehat{f}(x)$.

Before we begin any computation, let's clarify what is random and what is deterministic. The deterministic quantities are:

- the point $x$,
- the true function $f(x)$,
- the noise variance $\sigma^2$.

The random quantities are:

- the training dataset $\mathcal{D} := \{(X_i, Y_i)\}$,
- the learned predictor $\widehat{Y}$ (since it is a function of $\mathcal{D}$),
- the response $Y = f(x) + \epsilon$.

Therefore, all expectations will be taken over the joint randomness of the training set $\mathcal{D}$ and the noise $\epsilon$.

We can decompose the mean-squared error:
\[
\begin{aligned}
\text{MSE}(Y, \widehat{Y})
&= \mathbb{E}\bigl((Y - \widehat{Y})^2\bigr) \\
&= \mathbb{E}\left[(f(x) - \widehat{f}(x) - \epsilon)^2\right] \\
&= \mathbb{E}\bigl((f(x) - \widehat{f}(x))^2\bigr)
- 2\,\mathbb{E}(f(x) - \widehat{f}(x))\,\underbrace{\mathbb{E}(\epsilon)}_{=0}
+ \mathbb{E}(\epsilon^2) \\
&= \mathbb{E}\bigl((f(x) - \widehat{f}(x))^2\bigr) + \mathbb{E}(\epsilon^2).
\end{aligned}
\]

Above, we also crucially assumed that the noise $\epsilon$ is independent of the training data $\mathcal{D}$; otherwise, the cross-term does not vanish. Let us now proceed with the calculation.

Adding and subtracting $\mu(x) := \mathbb{E}(\widehat{f}(x))$ gives
\[
\begin{aligned}
\text{MSE}(Y, \widehat{Y})
&= \mathbb{E}\bigl((f(x) - \mu(x))^2\bigr)
+ \mathbb{E}\bigl((\widehat{f}(x) - \mu(x))^2\bigr) \\
&\quad + 2 (f(x) - \mu(x))\,\underbrace{\mathbb{E}(\mu(x) - \widehat{f}(x))}_{=0}
+ \sigma^2 \\
&= (f(x) - \mu(x))^2
+ \mathbb{E}\bigl((\widehat{f}(x) - \mathbb{E}(\widehat{f}(x)))^2\bigr)
+ \sigma^2 \\
&= \text{Bias}^2 + \text{Variance} + \sigma^2.
\end{aligned}
\]

This computation tells us that the *reducible* error of a machine learning model comes from two sources: the bias and the variance.

### The bias–variance trade-off

The bias–variance trade-off follows from the above decomposition and describes how these two factors are affected by model complexity.

> **The bias–variance trade-off**  
> As you increase the complexity of a model, the bias generally decreases while the variance increases.  
> Conversely, decreasing the model complexity tends to increase the bias while decreasing the variance.

This should be understood as a general principle rather than a rule. It is useful because it highlights that one must choose the model carefully for the problem at hand. It also provides a causal explanation for underfitting and overfitting, by attributing low model complexity to a high-bias, low-variance regime—and vice versa for high model complexity.

A key point of contention here concerns the word *complexity*. What exactly do we mean by complexity? Generally, model complexity refers to how rich a model class is, i.e. how many functions it can represent. There are precise ways to measure this (e.g. Rademacher complexity, covering numbers).

A tempting way to measure model complexity is by the *number of parameters* in the model. Although this is a reasonable answer for classical models, it is in fact a surprisingly bad metric in general. Indeed, the function
\[
f(a, b) = a \sin(bx)
\]
has just two parameters, yet $(a, b)$ can be tuned to interpolate any finite number of points. This gives it high bias on unseen data and high variance. Therefore, parameter count alone cannot serve as a meaningful measure of model complexity.

At this point, one might reasonably ask: if parameter count is not a good proxy for complexity, then why does the bias–variance trade-off work as well as it does in many classical settings?

The answer is that for many traditional models, increasing the number of parameters does monotonically increase the richness of the function class *in a controlled way*. Linear regression with polynomial features, splines with increasing knot counts, or $k$-nearest neighbours with decreasing $k$ all give rise to model families that become progressively more flexible, but also increasingly sensitive to noise. In these regimes, it is often empirically true that bias decreases while variance increases, leading to the familiar U-shaped test error curve.

However, modern machine learning models (e.g. deep neural networks) do not obey this simple picture.

### When the trade-off breaks down

In modern practice, models are frequently trained in the overparameterised regime, where the number of parameters is much greater than the number of training samples. Classical statistical intuition would predict catastrophic overfitting and extremely poor generalisation. Yet, in many cases, the opposite is observed: the test error continues to decrease as the model size grows.

This phenomenon is often referred to as *double descent*. As model complexity increases, the test error first follows the classical U-shaped curve, then peaks near the interpolation threshold (where the model fits the training data exactly), and finally decreases again as the model becomes even more overparameterised.

From the perspective of the bias–variance decomposition, this behaviour is unintuitive. Increasing model capacity beyond the point of perfect interpolation should, if anything, further increase variance. The key insight is that not all interpolating solutions are equally complex. The training algorithm itself plays a crucial role in selecting which function is learned.

### Implicit regularisation

In models such as neural networks trained with gradient-based methods, the optimisation procedure introduces an implicit regularisation bias. Even though the hypothesis class is extremely large, optimisation tends to favour solutions with particular structure: low-norm weights, smoother functions, or low-frequency representations. As a result, increasing the number of parameters does not necessarily increase the *effective* complexity of the learned predictor.

In other words, there is a distinction between:

- *expressive capacity*: what the model could represent in principle,
- *effective complexity*: what the model actually learns under a given training procedure.

The classical bias–variance trade-off implicitly assumes that these two notions coincide, but in modern settings they often do not.

### What remains useful

None of this means that the bias–variance trade-off is wrong. Rather, it is incomplete. The decomposition itself is an identity and always holds. What fails is the heuristic that increasing model size monotonically decreases bias and increases variance.

For practitioners, the key takeaway is that generalisation is governed not only by model architecture, but also by optimisation, data structure, and inductive bias. For theorists, this mismatch motivates much of the recent machine learning literature aimed at understanding generalisation more deeply.

In conclusion, the bias–variance trade-off is a valuable conceptual tool—but like most tools, it must be applied with care and with a clear understanding of its assumptions.

