<!-- markdownlint-disable -->

<a href="../flowfusion/symplectic_flow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `symplectic.py`

---

<a href="../flowfusion/symplectic_flow.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymplecticMLP`

```python
SymplecticMLP(
    n_data_dims,
    n_conditionals,
    embedding_dimensions,
    units,
    activation=nn.SiLU()
)
```

A specialized network that directly outputs a divergence-free velocity field by parameterizing the dynamics of position (q) and momentum (p) separately.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mlp_q_dynamics` | `torch.nn.Sequential` | Network for position dynamics. |
| `mlp_p_dynamics` | `torch.nn.Sequential` | Network for momentum dynamics. |
| `W` | `torch.Tensor` | Weights for time embedding. |

<a href="../flowfusion/symplectic_flow.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_data_dims,
    n_conditionals,
    embedding_dimensions,
    units,
    activation=nn.SiLU()
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_data_dims` | `int` | - | Dimension of inputs. |
| `n_conditionals` | `int` | - | Dimension of conditionals. |
| `embedding_dimensions` | `int` | - | Dimension of time embedding. |
| `units` | `list of int` | - | Number of hidden units per layer. |
| `activation` | `torch.nn.Module`, optional | `nn.SiLU()` | Activation function. |

---

<a href="../flowfusion/symplectic_flow.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `_create_mlp`

```python
_create_mlp(input_dim, output_dim, units, activation)
```

Initialises an MLP for modelling dynamics.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dim` | `int` | Dimension of inputs. |
| `output_dim` | `int` | Dimension of outputs. |
| `units` | `list of int` | Number of hidden units per layer. |
| `activation` | `torch.nn.Module` | Activation function. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `mlp` | `torch.nn.Sequential` | MLP. |

---

<a href="../flowfusion/symplectic_flow.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(t, state, conditional)
```

Evaluates Hamiltonian dynamics, (dq/dt, dp/dt).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | `torch.Tensor` | Time tensor. |
| `state` | `torch.Tensor` | Position and momentum. |
| `conditional` | `torch.Tensor` | Conditional inputs. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `v` | `torch.Tensor` | First half, position dynamics, dq/dt = dH/dp. Second half, momentum dynamics, dp/dt = -dH/dq |

### Notes

The symplectic structure ensures that the flow preserves phase space volume exactly, eliminating the need for expensive divergence computations during log probability evaluation.

---

<a href="../flowfusion/symplectic_flow.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SymplecticFlowModel`

```python
SymplecticFlowModel(
    model,
    shift,
    scale,
    conditional_shift,
    conditional_scale
)
```

The main model class. It uses a SymplecticMLP to define its dynamics, guaranteeing a fast sampler and enabling a fast, exact log_prob.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `flowfusion.symplectic_flow.SymplecticMLP` | MLPs for the dynamics. |
| `shift` | `torch.Tensor` | Input shift. |
| `scale` | `torch.Tensor` | Input scale. |
| `conditional_shift` | `torch.Tensor` | Conditional shift. |
| `conditional_scale` | `torch.Tensor` | Conditional scale. |

<a href="../flowfusion/symplectic_flow.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model,
    shift,
    scale,
    conditional_shift,
    conditional_scale
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `flowfusion.symplectic_flow.SymplecticMLP` | MLPs for the dynamics. |
| `shift` | `torch.Tensor` | Input shift. |
| `scale` | `torch.Tensor` | Input scale. |
| `conditional_shift` | `torch.Tensor` | Conditional shift. |
| `conditional_scale` | `torch.Tensor` | Conditional scale. |

---

<a href="../flowfusion/symplectic_flow.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample`

```python
sample(shape, conditional=None, num_steps=1)
```

Generate samples from the symplectic flow.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` | - | Desired sample shape to generate. |
| `conditional` | `torch.Tensor`, optional | `None` | Conditional inputs. |
| `num_steps` | `int`, optional | `1` | Number of time steps to use when integrating the flow. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `x0_final` | `torch.Tensor` | Samples from the target distribution, given `conditional`. |

### Notes

Due to the symplectic structure, samples can be generated with very few integration steps (often just 1) while maintaining high quality, making this approach much faster than traditional normalizing flows.

---

<a href="../flowfusion/symplectic_flow.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prob`

```python
log_prob(x, conditional=None, atol=1e-5, rtol=1e-5)
```

Compute the log probability of samples from the target.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | - | Samples from the target distribution. |
| `conditional` | `torch.Tensor`, optional | `None` | Conditional inputs. |
| `atol` | `float`, optional | `1e-5` | Absolute error tolerance for ODE solver. |
| `rtol` | `float`, optional | `1e-5` | Relative error tolerance for ODE solver. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `log_p_x` | `torch.Tensor` | Log probability of `x` given `conditional`. |

### Notes

The symplectic structure guarantees that the Jacobian determinant equals 1, allowing exact log probability computation without expensive trace estimation.

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._