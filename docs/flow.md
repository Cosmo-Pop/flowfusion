<!-- markdownlint-disable -->

<a href="../flowfusion/flow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `flow.py`

---

<a href="../flowfusion/flow.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODEFlow`

```python
ODEFlow(
    target_dimension=1,
    hidden_units=[128, 128],
    activation=nn.SiLU,
    target_shift=None,
    target_scale=None
)
```

ODE Flow model.

This class implements a conditional ODE flow using a neural network to model the dynamics. It includes methods for computing the dynamics, sampling, and calculating the flow matching loss.

Note that we define the ODE transform from a base unit normal (at t=T) to a target distribution (at t=0), i.e., the forward transform integrates backwards in time, consistent with diffusion models.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_dimension` | `int` | Dimension of the target distribution. |
| `layers` | `torch.nn.ModuleList` | List of neural network layers. |
| `velocity` | `torch.nn.Sequential` | Model for the dynamics of the flow. |
| `target_shift` | `torch.Tensor` | Shift to be applied to the target distribution. |
| `target_scale` | `torch.Tensor` | Scale to be applied to the target distribution. |
| `twopi` | `torch.Tensor` | Tensor containing 2*pi. |

<a href="../flowfusion/flow.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    target_dimension=1,
    hidden_units=[128, 128],
    activation=nn.SiLU,
    target_shift=None,
    target_scale=None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dimension` | `int`, optional | `1` | Dimension of the target distribution. |
| `hidden_units` | `list of int`, optional | `[128, 128]` | Number of hidden units per layer in the dynamics network. |
| `activation` | `torch.nn.Module`, optional | `nn.SiLU` | Activation function. |
| `target_shift` | `torch.Tensor`, optional | `None` | Shift to be applied to the target distribution. |
| `target_scale` | `torch.Tensor`, optional | `None` | Scale to be applied to the target distribution. |

---

<a href="../flowfusion/flow.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamics`

```python
dynamics(t, states)
```

Computes the dynamics, dx/dt, of the ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `states` | `tuple of torch.Tensor` | Tuple containing the current state. Current state (x) is assumed to be in `states[0]`. Note that x here is assumed to be normalized by the shift and scale. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dxdt` | `torch.Tensor` | The velocity field. |

---

<a href="../flowfusion/flow.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamics_with_jacobian`

```python
dynamics_with_jacobian(t, states)
```

Computes the dynamics of the ODE flow and the log determinant of the Jacobian. These correspond to dx/dt and dp(x)/dt.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `states` | `tuple of torch.Tensor` | Tuple containing the current state, and log Jacobian. Current state (x) is assumed to be in `states[0]`. Log Jacobian assumed to be in `states[1]`. Note that x here is assumed to be normalized by the shift and scale. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dxdt` | `torch.Tensor` | The velocity field. |
| `divergence` | `torch.Tensor` | Divergence of the velocity field. |

---

<a href="../flowfusion/flow.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(t, states)
```

Computes the forward pass of the ODE flow, i.e., the velocity field.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `states` | `tuple of torch.Tensor` | Tuple containing the current state. Current state (x) is assumed to be in `states[0]`. Note that x here is assumed to be normalized by the shift and scale. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dxdt` | `torch.Tensor` | The velocity field. |

---

<a href="../flowfusion/flow.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_linear_velocity_field`

```python
compute_linear_velocity_field(x0, xT, t)
```

Computes the ideal linear velocity field for the ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x0` | `torch.Tensor` | Initial state at t=0 (observed samples). |
| `xT` | `torch.Tensor` | Final state at t=T (base density). |
| `t` | `torch.Tensor` | Time tensor. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `xt` | `torch.Tensor` | Interpolated state at times t. |
| `v_hat` | `torch.Tensor` | Ideal velocity (dx/dt). |

---

<a href="../flowfusion/flow.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flow_matching_loss`

```python
flow_matching_loss(x)
```

Computes the flow matching loss for the ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Initial state at t=0 (observed samples). |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `loss` | `torch.Tensor` | Flow matching loss. |

---

<a href="../flowfusion/flow.py#L259"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample`

```python
sample(xT, gradients=False)
```

This function transforms base samples to the target under the ODE flow via odeint, by integrating backwards in time from t=T to t=0.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xT` | `torch.Tensor` | - | Final state at t=T (base density). |
| `gradients` | `bool`, optional | `False` | Whether to compute in a `torch.no_grad()` context. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x0` | `torch.Tensor` | Transformed samples from target density at t=0. |

---

<a href="../flowfusion/flow.py#L308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve_ode_forward`

```python
solve_ode_forward(
    x,
    atol=1e-5,
    rtol=1e-5,
    method="dopri5",
    options=None,
    adjoint=False
)
```

This solves the pair of ODEs forward in time (t=0 (target) -> t=T (base)) to find the base x(t=T) samples and log probabilities associated with some input x(t=0) samples.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | - | The initial state x(t=0) (observed samples). Note this is assumed to be normalized by the shift and scale. Shape: (batch_size, n_covariates) |
| `atol` | `float`, optional | `1e-5` | Absolute error tolerance for the ODE solver. |
| `rtol` | `float`, optional | `1e-5` | Relative error tolerance for the ODE solver. |
| `method` | `str`, optional | `"dopri5"` | The ODE solver method to use. Must be a valid `torchdiffeq` option. |
| `options` | `dict`, optional | `None` | Additional options for the ODE solver. |
| `adjoint` | `bool`, optional | `False` | Whether to use the adjoint ODE solver. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `xT` | `torch.Tensor` | State x(t=T), i.e., samples from the base density. |
| `lp` | `torch.Tensor` | Log probabilities of input samples x(t=0). |

---

<a href="../flowfusion/flow.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prob`

```python
log_prob(
    x,
    atol=1e-5,
    rtol=1e-5,
    method="dopri5",
    options=None,
    adjoint=False
)
```

Computes the log probability of the input samples under the ODE flow. This includes the correction for the probabilities under the base density.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | - | Input samples, x(t=0). |
| `atol` | `float`, optional | `1e-5` | Absolute tolerance for the ODE solver. |
| `rtol` | `float`, optional | `1e-5` | Relative tolerance for the ODE solver. |
| `method` | `str`, optional | `"dopri5"` | ODE solver method. Must be a valid `torchdiffeq` option. |
| `options` | `dict`, optional | `None` | Additional options for the ODE solver. |
| `adjoint` | `bool`, optional | `False` | Whether to use the adjoint ODE solver. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `log_prob` | `torch.Tensor` | Log probability of the input samples. |

---

<a href="../flowfusion/flow.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConditionalODEFlow`

```python
ConditionalODEFlow(
    target_dimension=1,
    conditional_dimension=1,
    hidden_units=[128, 128],
    activation=nn.SiLU,
    target_shift=None,
    target_scale=None,
    conditional_shift=None,
    conditional_scale=None
)
```

This class implements a conditional ODE flow using a neural network to model the dynamics. It includes methods for computing the dynamics, sampling, and calculating the flow matching loss.

Note that we define the ODE transform from a base unit normal (at t=T) to a target distribution (at t=0), i.e., the forward transform integrates backwards in time, consistent with diffusion models.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_dimension` | `int` | Dimension of the target distribution. |
| `conditional_dimension` | `int` | Dimension of the conditional inputs. |
| `layers` | `torch.nn.ModuleList` | List of neural network layers. |
| `velocity` | `torch.nn.Sequential` | Model for the dynamics of the flow. |
| `target_shift` | `torch.Tensor` | Shift to be applied to the target distribution. |
| `target_scale` | `torch.Tensor` | Scale to be applied to the target distribution. |
| `conditional_shift` | `torch.Tensor` | Shift to be applied to the conditional inputs. |
| `conditional_scale` | `torch.Tensor` | Scale to be applied to the conditional inputs. |
| `twopi` | `torch.Tensor` | Tensor containing 2*pi. |

<a href="../flowfusion/flow.py#L473"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    target_dimension=1,
    conditional_dimension=1,
    hidden_units=[128, 128],
    activation=nn.SiLU,
    target_shift=None,
    target_scale=None,
    conditional_shift=None,
    conditional_scale=None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dimension` | `int`, optional | `1` | Dimension of the target distribution. |
| `conditional_dimension` | `int`, optional | `1` | Dimension of the conditional inputs. |
| `hidden_units` | `list of int`, optional | `[128, 128]` | Number of hidden units per layer in the dynamics network. |
| `activation` | `torch.nn.Module`, optional | `nn.SiLU` | Activation function. |
| `target_shift` | `torch.Tensor`, optional | `None` | Shift to be applied to the target distribution. |
| `target_scale` | `torch.Tensor`, optional | `None` | Scale to be applied to the target distribution. |
| `conditional_shift` | `torch.Tensor`, optional | `None` | Shift to be applied to the conditional inputs. |
| `conditional_scale` | `torch.Tensor`, optional | `None` | Scale to be applied to the conditional inputs. |

---

<a href="../flowfusion/flow.py#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamics`

```python
dynamics(t, states)
```

Computes the dynamics, dx/dt,of the ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `states` | `tuple of torch.Tensor` | Tuple containing the current state and conditional inputs. Current state (x) is assumed to be in `states[0]`. Conditional inputs are assumed to be in `states[1]`. Note that x here is assumed to be normalized by the shift and scale. Note that the conditionals get normalized inside the function. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dxdt` | `torch.Tensor` | The velocity field. |
| `zeros` | `torch.Tensor` | Zero gradient for the conditionals. |

---

<a href="../flowfusion/flow.py#L598"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamics_with_jacobian`

```python
dynamics_with_jacobian(t, states)
```

Computes the dynamics, dx/dt, of the ODE flow and the log determinant of the Jacobian, dp(x)/dt.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `states` | `tuple of torch.Tensor` | Tuple containing the current state, conditional inputs, and log Jacobian. Current state (x) is assumed to be in `states[0]`. Conditional inputs are assumed to be in `states[1]`. Log Jacobian assumed to be in `states[2]`. Note that x here is assumed to be normalized by the shift and scale. Note that the conditionals get normalized inside the function. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dxdt` | `torch.Tensor` | The velocity field. |
| `zeros` | `torch.Tensor` | Zero gradient for the conditionals. |
| `divergence` | `torch.Tensor` | Divergence of the velocity field. |

---

<a href="../flowfusion/flow.py#L655"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(t, states)
```

Computes the forward pass of the conditional ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `states` | `tuple of torch.Tensor` | Tuple containing the current state and conditional inputs. Current state (x) is assumed to be in `states[0]`. Conditional inputs are assumed to be in `states[1]`. Note that x here is assumed to be normalized by the shift and scale. Note that the conditionals get normalized inside the function. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `dxdt` | `torch.Tensor` | The velocity field. |

---

<a href="../flowfusion/flow.py#L679"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_linear_velocity_field`

```python
compute_linear_velocity_field(x0, xT, t)
```

Computes the ideal linear velocity field for the ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x0` | `torch.Tensor` | Initial state at t=0 (observed samples). |
| `xT` | `torch.Tensor` | Final state at t=T (base density). |
| `t` | `torch.Tensor` | Time tensor. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `xt` | `torch.Tensor` | Interpolated state at times t. |
| `v_hat` | `torch.Tensor` | Ideal velocity (dx/dt). |

---

<a href="../flowfusion/flow.py#L711"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flow_matching_loss`

```python
flow_matching_loss(x, conditional)
```

Computes the flow matching loss for the ODE flow.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Initial state at t=0 (observed samples). |
| `conditional` | `torch.Tensor` | Conditional inputs. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `loss` | `torch.Tensor` | Flow matching loss. |

---

<a href="../flowfusion/flow.py#L750"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample`

```python
sample(xT, conditional, gradients=False)
```

This function transforms base samples to the target under the ODE flow via odeint, by integrating backwards in time from t=T to t=0.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xT` | `torch.Tensor` | - | Final state at t=T (base density). |
| `conditional` | `torch.Tensor` | - | Conditional inputs. |
| `gradients` | `bool`, optional | `False` | Whether to compute in a `torch.no_grad()` context. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x0` | `torch.Tensor` | Transformed samples from target density at t=0. |

---

<a href="../flowfusion/flow.py#L801"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `solve_ode_forward`

```python
solve_ode_forward(
    x,
    conditional,
    atol=1e-5,
    rtol=1e-5,
    method="dopri5",
    options=None,
    adjoint=False
)
```

This solves the pair of ODEs forward in time (t=0 (target) -> t=T (base)) to find the base x(t=T) samples and log probabilities associated with some input x(t=0) samples given the conditional inputs.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | - | The initial state x(t=0) (observed samples). Note this is assumed to be normalized by the shift and scale. Shape: (batch_size, n_covariates) |
| `conditional` | `torch.Tensor` | - | Conditional inputs. Shape: (batch_size, n_conditionals) |
| `atol` | `float`, optional | `1e-5` | Absolute error tolerance for the ODE solver. |
| `rtol` | `float`, optional | `1e-5` | Relative error tolerance for the ODE solver. |
| `method` | `str`, optional | `"dopri5"` | The ODE solver method to use. Must be a valid `torchdiffeq` option. |
| `options` | `dict`, optional | `None` | Additional options for the ODE solver. |
| `adjoint` | `bool`, optional | `False` | Whether to use the adjoint ODE solver. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `xT` | `torch.Tensor` | State x(t=T), i.e., samples from the base density. |
| `lp` | `torch.Tensor` | Log probabilities of input samples x(t=0). |

---

<a href="../flowfusion/flow.py#L885"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prob`

```python
log_prob(
    x,
    conditional,
    atol=1e-5,
    rtol=1e-5,
    method="dopri5",
    options=None,
    adjoint=False
)
```

Computes the conditional log probability of the input samples under the ODE flow. This includes the correction for the probabilities under the base density.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | - | Input samples, x(t=0). |
| `conditional` | `torch.Tensor` | - | Conditional inputs. |
| `atol` | `float`, optional | `1e-5` | Absolute tolerance for the ODE solver. |
| `rtol` | `float`, optional | `1e-5` | Relative tolerance for the ODE solver. |
| `method` | `str`, optional | `"dopri5"` | ODE solver method. Must be a valid `torchdiffeq` option. |
| `options` | `dict`, optional | `None` | Additional options for the ODE solver. |
| `adjoint` | `bool`, optional | `False` | Whether to use the adjoint ODE solver. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `log_prob` | `torch.Tensor` | Log probability of the input samples. |

### See Also

- `ODEFlow` : Unconditional normalizing flow implementation

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._