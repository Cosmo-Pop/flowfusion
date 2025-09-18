<!-- markdownlint-disable -->

<a href="../flowfusion/diffusion.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `diffusion.py`

---

<a href="../flowfusion/diffusion.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MLP`

```python
MLP(
    n_dimensions=2,
    n_conditionals=1,
    embedding_dimensions=8,
    units=[128],
    activation=torch.nn.SiLU(),
    sigma_initialization=16
)
```

Multilayer perceptron for learning the score function.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_dimensions` | `int` | Number of input/output dimensions |
| `n_conditionals` | `int` | Number of conditional inputs |
| `n_layers` | `int` | Number of hidden layers |
| `architecture` | `list of int` | Network architecture (input/output dims of each layer) |
| `activation` | `torch.nn.Module` | Activation function |
| `NN` | `torch.nn.ModuleList` | List of network layers |
| `W` | `torch.nn.Parameter` | Weights for the time embedding |
| `pi` | `torch.Tensor` | Tensor containing pi |

<a href="../flowfusion/diffusion.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_dimensions=2,
    n_conditionals=1,
    embedding_dimensions=8,
    units=[128],
    activation=torch.nn.SiLU(),
    sigma_initialization=16
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_dimensions` | `int`, optional | `2` | Number of input/output dimensions |
| `n_conditionals` | `int`, optional | `1` | Number of conditional inputs |
| `embedding_dimensions` | `int`, optional | `8` | Number of dimensions of the time embedding |
| `units` | `list of int`, optional | `[128]` | Number of hidden units per layer |
| `activation` | `torch.nn.Module`, optional | `torch.nn.SiLU()` | Torch activation function |
| `sigma_initialization` | `float`, optional | `16` | Standard deviation used to generate initial embedding weights |

---

<a href="../flowfusion/diffusion.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(t, x, conditional=None)
```

Forward call to the MLP

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Times to evaluate the score network at |
| `x` | `torch.Tensor` | Inputs to evaluate the score network at |
| `conditional` | `torch.Tensor`, optional | Conditional inputs |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x` | `torch.Tensor` | Outputs from the network |

---

<a href="../flowfusion/diffusion.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ScoreModel`

```python
ScoreModel(
    model=None,
    sde=None,
    conditional=None,
    no_sigma=False,
    hutchinson=False
)
```

Score-based generative model that learns the score function of any given data distribution and generates samples by reversing an ODE/SDE

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Score model. Usually an `MLP`. |
| `sde` | `torch.nn.Module` | Stochastic differential equation. Usually a `VPSDE`, `VESDE` or `SUBVPSDE`. |
| `conditional` | `torch.Tensor` | Internal variable for tracking conditional inputs. |
| `no_sigma` | `bool` | If `True`, `model` is assumed to return score(x, t, conditional). If `False`, `model` is assumed to return score(x, t, conditional) * sigma(t). |
| `prob` | `bool` | Internal variable to track whether the trace of the Jacobian is included in the forward call (automatically set/reset when calling `solve_odes_forward`). |
| `hutch` | `bool` | Internal variable to track whether the Skilling--Hutchinson trace estimator is used in `solve_odes_forward`. |

<a href="../flowfusion/diffusion.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model=None,
    sde=None,
    conditional=None,
    no_sigma=False,
    hutchinson=False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module`, optional | `None` | Score model. Usually an `MLP`. |
| `sde` | `torch.nn.Module`, optional | `None` | Stochastic differential equation. Usually a `VPSDE`, `VESDE` or `SUBVPSDE`. |
| `conditional` | `torch.Tensor`, optional | `None` | Initial value of conditioning variable (can be updated) |
| `no_sigma` | `bool`, optional | `False` | If `True`, `model` is assumed to return score(x, t, conditional). If `False`, `model` is assumed to return score(x, t, conditional) * sigma(t). |
| `hutchinson` | `bool`, optional | `False` | If `True`, `solve_odes_forward` will be computed using the Skilling--Hutchinson trace estimator. |

---

<a href="../flowfusion/diffusion.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `score`

```python
score(t, x, conditional=None)
```

Compute the time dependent score.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Times to compute score at |
| `x` | `torch.Tensor` | Inputs to evaluate score at |
| `conditional` | `torch.Tensor`, optional | Conditional inputs |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `score` | `torch.Tensor` | Score computed via a forward pass through the score network. |

---

<a href="../flowfusion/diffusion.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `loss_fn`

```python
loss_fn(x, conditional=None)
```

Denoising score matching loss.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Inputs |
| `conditional` | `torch.Tensor`, optional | Conditional inputs |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `loss` | `torch.Tensor` | Denoising score matching loss |

---

<a href="../flowfusion/diffusion.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ode_drift`

```python
ode_drift(t, x, conditional=None)
```

Drift term in the probability flow ODE

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Times |
| `x` | `torch.Tensor` | Inputs |
| `conditional` | `torch.Tensor`, optional | Conditional inputs |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `f_tilde` | `torch.Tensor` | ODE drift |

---

<a href="../flowfusion/diffusion.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(t, states)
```

Compute dx/d, and optionally dp(x)/dt at t. Input to the ODE solver.

If `self.hutch` is `True`, the Skilling--Hutchinson trace estimator will be used to compute dp(x)/dt.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Current times. |
| `states` | `torch.Tensor` | Current states. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x_dot` | `torch.Tensor` | Time derivative of x. |
| `divergence` | `torch.Tensor`, optional | Time derivative of p(x). Only returned if `self.prob` is `True`. |

### See Also

- `sample_ode_from_base` : Integrates dx/dt to generate samples.
- `solve_odes_forward` : Integrates dx/dt and dp(x)/dt to compute log prob.

---

<a href="../flowfusion/diffusion.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample_sde`

```python
sample_sde(shape, conditional=None, steps=100)
```

An Euler-Maruyama integration of the model SDE backwards in time from t=T to t=0.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` | - | Shape of inputs/outputs. |
| `conditional` | `torch.Tensor`, optional | `None` | Conditional inputs. |
| `steps` | `int`, optional | `100` | Number of timesteps in SDE solution. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x` | `torch.Tensor` | Samples from the model. |

---

<a href="../flowfusion/diffusion.py#L377"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample_ode_from_base`

```python
sample_ode_from_base(
    base_samples,
    conditional=None,
    atol=1e-4,
    rtol=1e-4,
    method="dopri5",
    options=None
)
```

Generate samples deterministically by solving ODE backwards in time from t=T to t=0.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_samples` | `torch.Tensor` | - | Base samples to transform, i.e., x(t=T) ~ N(0,1). |
| `conditional` | `torch.Tensor`, optional | `None` | Conditional inputs. |
| `atol` | `float`, optional | `1e-4` | Absolute error tolerance for ODE solver. |
| `rtol` | `float`, optional | `1e-4` | Relative error tolerance for ODE solver. |
| `method` | `string`, optional | `"dopri5"` | ODE solving routine. |
| `options` | `dict`, optional | `None` | Dictionary of additional ODE solver options. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x0_samples` | `torch.Tensor` | Samples in parameter space generated by transforming `base_samples`. |

### See Also

- `torchdiffeq.odeint` : ODE solver used (including option definitions)
- `torchdiffeq.odeint_adjoint` : ODE solver used when backward pass needed

---

<a href="../flowfusion/diffusion.py#L453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve_odes_forward`

```python
solve_odes_forward(
    x0_samples,
    conditional=None,
    atol=1e-5,
    rtol=1e-5,
    method="dopri5",
    options=None
)
```

This solves the pair of ODEs forward in time to find the base samples, x(t=T), and log probabilities associated with some input samples, x(t=0).

Integrates from t=0 to t=T.

If `self.hutch` is `True`, the Skilling--Hutchinson trace estimator will be used in the integrand.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x0_samples` | `torch.Tensor` | - | Samples in parameter space, i.e., x(t=0). |
| `conditional` | `torch.Tensor`, optional | `None` | Conditional inputs. |
| `atol` | `float`, optional | `1e-5` | Absolute error tolerance for ODE solver. |
| `rtol` | `float`, optional | `1e-5` | Relative error tolerance for ODE solver. |
| `method` | `string`, optional | `"dopri5"` | ODE solving routine. |
| `options` | `dict`, optional | `None` | Dictionary of additional ODE solver options. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `base_samples` | `torch.Tensor` | Samples from the base density, i.e., x(t=T). |
| `log_prob` | `torch.Tensor` | Log probability density of `x0_samples`, i.e. p[x(t=0)] - p[x(t=T)]. |

### See Also

- `sample_ode_from_base` : Solves in the opposite direction (from t=T to t=0).

---

<a href="../flowfusion/diffusion.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VESDE`

```python
VESDE(sigma_min=1e-2, sigma_max=10.0, T=1.0, epsilon=1e-5)
```

Variance Exploding SDE.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `T` | `torch.Tensor` | Maximum integration time of stochastic process. |
| `epsilon` | `torch.Tensor` | Minimum integration time of stochastic process. |
| `sigma_max` | `torch.Tensor` | Marginal standard deviation at t=T. |
| `sigma_min` | - | Marginal standard deviation at t=0. |

<a href="../flowfusion/diffusion.py#L555"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(sigma_min=1e-2, sigma_max=10.0, T=1.0, epsilon=1e-5)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma_min` | `float`, optional | `1e-2` | Marginal standard deviation at t=0. |
| `sigma_max` | `float`, optional | `10.0` | Marginal standard deviation at t=T. |
| `T` | `float`, optional | `1.0` | Maximum integration time. |
| `epsilon` | `float`, optional | `1e-5` | Minimum integration time. |

---

<a href="../flowfusion/diffusion.py#L574"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sigma`

```python
sigma(t)
```

Marginal standard deviation at time t, sigma(t).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time to compute sigma(t) at. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `sigma` | `torch.Tensor` | Marginal standard deviation, sigma(t). |

---

<a href="../flowfusion/diffusion.py#L590"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `diffusion`

```python
diffusion(t, x)
```

Diffusion term in the forward SDE, g(t).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Times. |
| `x` | `torch.Tensor` | States (not used). |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `g` | `torch.Tensor` | SDE diffusion, g(t). |

---

<a href="../flowfusion/diffusion.py#L611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `drift`

```python
drift(t, x)
```

Drift term in forward SDE, f(x,t)=0.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Times. |
| `x` | `torch.Tensor` | States. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `f` | `torch.Tensor` | SDE drift, f(x,t). |

---

<a href="../flowfusion/diffusion.py#L705"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prior`

```python
prior(shape, mu=None)
```

Prior distribution.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple` | Dimensions of distribution. |
| `mu` | `torch.Tensor`, optional | Prior mean. Not recommended to set this explicitly. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| - | `torch.distributions.normal.Normal` | Gaussian prior. |

---

<a href="../flowfusion/diffusion.py#L728"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VPSDE`

```python
VPSDE(beta_min=0.1, beta_max=20, T=1.0, epsilon=1e-3)
```

Variance Preserving SDE.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `T` | `torch.Tensor` | Maximum integration time of stochastic process. |
| `epsilon` | `torch.Tensor` | Minimum integration time of stochastic process. |
| `beta_max` | `torch.Tensor` | Beta at t=T. |
| `beta_min` | `torch.Tensor` | Beta at t=0. |

<a href="../flowfusion/diffusion.py#L744"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(beta_min=0.1, beta_max=20, T=1.0, epsilon=1e-3)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta_min` | `float`, optional | `0.1` | Beta at t=0. |
| `beta_max` | `float`, optional | `20` | Beta at t=T. |
| `T` | `float`, optional | `1.0` | Maximum integration time of stochastic process. |
| `epsilon` | `float`, optional | `1e-3` | Minimum integration time of stochastic process. |

---

<a href="../flowfusion/diffusion.py#L769"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `beta`

```python
beta(t)
```

Computes beta(t).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Times. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `beta` | `torch.Tensor` | Coefficient beta(t). |

---

<a href="../flowfusion/diffusion.py#L905"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SUBVPSDE`

```python
SUBVPSDE(beta_min=0.1, beta_max=20, T=1.0, epsilon=1e-5)
```

Sub-Variance Preserving SDE.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `T` | `torch.Tensor` | Maximum integration time of stochastic process. |
| `epsilon` | `torch.Tensor` | Minimum integration time of stochastic process. |
| `beta_max` | `torch.Tensor` | Beta at t=T. |
| `beta_min` | `torch.Tensor` | Beta at t=0. |

---

<a href="../flowfusion/diffusion.py#L1091"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `denoising_score_matching`

```python
denoising_score_matching(score_model, x, conditional=None)
```

Denoising score matching loss function.
Based on Song et al. (2020; arXiv:2011.13456).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `score_model` | `flowfusion.diffusion.ScoreModel` | Class containing the score model. |
| `x` | `torch.Tensor` | Inputs drawn from the target distribution. |
| `conditional` | `torch.Tensor`, optional | Conditional inputs. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `loss` | `torch.Tensor` | Denoising score matching loss. |

---

<a href="../flowfusion/diffusion.py#L1139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `log_prob_score_matching`

```python
log_prob_score_matching(score_model, x, conditional=None)
```

Score matching loss function with likelihood weighting.
Based on Song et al. (2020; arXiv:2101.09258).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `score_model` | `flowfusion.diffusion.ScoreModel` | Class containing the score model. |
| `x` | `torch.Tensor` | Inputs drawn from the target distribution. |
| `conditional` | `torch.Tensor`, optional | Conditional inputs. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `loss` | `torch.Tensor` | Score matching loss with likelihood weighting. |

---

<a href="../flowfusion/diffusion.py#L1188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PopulationModelDiffusion`

```python
PopulationModelDiffusion(
    model=None,
    sde=None,
    shift=None,
    scale=None,
    method="dopri5",
    no_sigma=False,
    hutchinson=False,
    options=None
)
```

Diffusion model class without conditionals.

This class wraps a `ScoreModel` to provide useful functionality for population modelling and unconditional density estimation.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `flowfusion.diffusion.MLP` | Score network. |
| `sde` | `torch.nn.Module` | SDE class. Usually `VESDE`, `VPSDE`, or `SUBVPSDE`. |
| `score_model` | `flowfusion.diffusion.ScoreModel` | Score model class. |
| `shift` | `torch.Tensor` | Parameter shift for inputs/outputs. |
| `scale` | `torch.Tensor` | Parameter scale for inputs/outputs. |
| `method` | `str` | Name of ODE solver. |
| `options` | `dict` | Optional arguments for ODE solver. |

### See Also

- `PopulationModelDiffusionConditional` : Similar class for conditional densities.
- `ScoreModel` : Underlying class for the score-based model.
- `MLP` : Underlying class for the score network.

---

<a href="../flowfusion/diffusion.py#L1365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PopulationModelDiffusionConditional`

```python
PopulationModelDiffusionConditional(
    model=None,
    sde=None,
    shift=None,
    scale=None,
    conditional_shift=None,
    conditional_scale=None,
    no_sigma=False,
    method="dopri5",
    options=None
)
```

Diffusion model class with conditionals.

This class wraps a `ScoreModel` to provide useful functionality for population modelling and conditional density estimation.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `flowfusion.diffusion.MLP` | Score network. |
| `sde` | `torch.nn.Module` | SDE class. Usually `VESDE`, `VPSDE`, or `SUBVPSDE`. |
| `score_model` | `flowfusion.diffusion.ScoreModel` | Score model class. |
| `shift` | `torch.Tensor` | Parameter shift for inputs/outputs. |
| `scale` | `torch.Tensor` | Parameter scale for inputs/outputs. |
| `conditional_shift` | `torch.Tensor` | Parameter shift for conditional inputs. |
| `conditional_scale` | `torch.Tensor` | Parameter scale for conditional inputs. |
| `method` | `str` | Name of ODE solver. |
| `options` | `dict` | Optional arguments for ODE solver. |

### See Also

- `PopulationModelDiffusion` : Similar class for unconditional densities.
- `ScoreModel` : Underlying class for the score-based model.
- `MLP` : Underlying class for the score network.

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._