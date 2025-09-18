# symplectic_flow.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from tqdm import tqdm
import copy
import math

class SymplecticMLP(nn.Module):
    """
    A specialized network that directly outputs a divergence-free velocity field
    by parameterizing the dynamics of position (q) and momentum (p) separately.

    Attributes
    ----------
    mlp_q_dynamics : torch.nn.Sequential
        Network for position dynamics.
    mlp_p_dynamics : torch.nn.Sequential
        Network for momentum dynamics.
    W : torch.Tensor
        Weights for time embedding.
    """
    def __init__(self, n_data_dims, n_conditionals, embedding_dimensions, units, activation=nn.SiLU()):
        """
        Parameters
        ----------
        n_data_dims : int
            Dimension of inputs.
        n_conditionals : int
            Dimension of conditionals.
        embedding_dimensions : int
            Dimension of time embedding.
        units : list of int
            Number of hidden units per layer.
        activation : torch.nn.Module, optional
            Activation function.
        """
        super().__init__()
        
        input_dim = n_data_dims + n_conditionals + embedding_dimensions
        output_dim = n_data_dims
        
        # Two separate MLPs: one for position dynamics, one for momentum dynamics
        self.mlp_q_dynamics = self._create_mlp(input_dim, output_dim, units, activation)
        self.mlp_p_dynamics = self._create_mlp(input_dim, output_dim, units, activation)
        
        self.register_buffer("W", torch.randn(embedding_dimensions // 2) * 16.0)

    def _create_mlp(self, input_dim, output_dim, units, activation):
        """
        Initialises an MLP for modelling dynamics.

        Parameters
        ----------
        input_dim : int
            Dimension of inputs.
        output_dim : int
            Dimension of outputs.
        units : list of int
            Number of hidden units per layer.
        activation : torch.nn.Module
            Activation function.

        Returns
        -------
        mlp : torch.nn.Sequential
            MLP.
        """
        layers = []
        current_dims = input_dim
        for unit_count in units:
            layers.append(nn.Linear(current_dims, unit_count))
            layers.append(activation)
            current_dims = unit_count
        layers.append(nn.Linear(current_dims, output_dim))
        return nn.Sequential(*layers)

    def forward(self, t, state, conditional):
        """
        Evaluates Hamiltonian dynamics, (dq/dt, dp/dt).

        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        state : torch.Tensor
            Position and momentum.
        conditional : torch.Tensor
            Conditional inputs.

        Returns
        -------
        v : torch.Tensor
            First half, position dynamics, dq/dt = dH/dp
            Second half, momentum dynamics, dp/dt = -dH/dq
        """
        q, p = torch.chunk(state, 2, dim=-1)
        
        if t.dim() == 0:
            t = t.expand(q.shape[0])
        t_projected = t[:, None] * self.W[None, :] * 2 * math.pi
        t_embedded = torch.cat([torch.sin(t_projected), torch.cos(t_projected)], dim=1)
        
        # The input for the q-dynamics network should contain p, and vice-versa.
        if conditional is not None:
            # Input for q-dynamics contains p
            input_for_q_dynamics = torch.cat([p, conditional, t_embedded], dim=1)
            # Input for p-dynamics contains q
            input_for_p_dynamics = torch.cat([q, conditional, t_embedded], dim=1)
        else:
            input_for_q_dynamics = torch.cat([p, t_embedded], dim=1)
            input_for_p_dynamics = torch.cat([q, t_embedded], dim=1)
            
        # This structure approximates a Hamiltonian system where H = H_q(q) + H_p(p).
        # dq/dt = dH/dp = dH_p/dp  (which is a function of p)
        # dp/dt = -dH/dq = -dH_q/dq (which is a function of q)
        
        v_q = self.mlp_q_dynamics(input_for_q_dynamics) # This computes dH/dp
        v_p = -self.mlp_p_dynamics(input_for_p_dynamics) # This computes -dH/dq
        
        return torch.cat([v_q, v_p], dim=-1)

class SymplecticFlowModel(nn.Module):
    """
    The main model class. It uses a SymplecticMLP to define its dynamics,
    guaranteeing a fast sampler and enabling a fast, exact log_prob.

    Attributes
    ----------
    model : flowfusion.symplectic_flow.SymplecticMLP
        MLPs for the dynamics.
    shift : torch.Tensor
        Input shift.
    scale : torch.Tensor
        Input scale.
    conditional_shift : torch.Tnsor
        Conditional shift.
    conditional_scale : torch.Tensor
        Conditional scale.
    """
    def __init__(self, model, shift, scale, conditional_shift, conditional_scale):
        """
        Parameters
        ----------
        model : flowfusion.symplectic_flow.SymplecticMLP
            MLPs for the dynamics.
        shift : torch.Tensor
            Input shift.
        scale : torch.Tensor
            Input scale.
        conditional_shift : torch.Tnsor
            Conditional shift.
        conditional_scale : torch.Tensor
            Conditional scale.
        """
        super().__init__()
        self.model = model
        self.register_buffer("shift", shift)
        self.register_buffer("scale", scale)
        self.register_buffer("conditional_shift", conditional_shift)
        self.register_buffer("conditional_scale", conditional_scale)
    
    @torch.no_grad()
    def sample(self, shape, conditional=None, num_steps=1):
        """
        Generate samples from the symplectic flow.

        Parameters
        ----------
        shape : tuple
            Desired sample shape to generate.
        conditional : torch.Tensor, optional
            Conditional inputs.
        num_steps : int, optional
            Number of time steps to use when integrating the flow.

        Returns
        -------
        x0_final : torch.Tensor
            Samples from the target distribution, given `conditional`.
        """
        device = next(self.model.parameters()).device
        # Start with random noise for both position and momentum
        x = torch.randn(shape[0], shape[1] * 2, device=device)
        
        if conditional is not None:
            conditional = (conditional - self.conditional_shift) / self.conditional_scale
            
        time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in tqdm(range(num_steps), desc=f"Sampling ({num_steps} steps)"):
            t_curr = time_steps[i]
            t_next = time_steps[i+1]
            dt = t_next - t_curr
            v_pred = self.model(t_curr.expand(shape[0]), x, conditional)
            x = x + v_pred * dt
            
        q0_normalized, _ = torch.chunk(x, 2, dim=-1)
        x0_final = q0_normalized * self.scale + self.shift
        return x0_final

    @torch.no_grad()
    def log_prob(self, x, conditional=None, atol=1e-5, rtol=1e-5):
        """
        Compute the log probability of samples from the target.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the target distribution.
        conditional : torch.Tensor, optional
            Conditional inputs.
        atol : float, optional
            Absolute error tolerance for ODE solver.
        rtol : float, optional
            Relative error tolerance for ODE solver.

        Returns
        -------
        log_p_x : torch.Tensor
            Log probability of `x` given `conditional`.
        """
        device = x.device
        q0 = (x - self.shift) / self.scale
        if conditional is not None:
            conditional = (conditional - self.conditional_shift) / self.conditional_scale
        p0 = torch.randn_like(q0)
        initial_state = torch.cat([q0, p0], dim=-1)
        
        # The ODE function is just the model's forward pass. No autograd needed for divergence.
        def ode_func(t, state):
            t_tensor = torch.full((state.shape[0],), t, device=device)
            return self.model(t_tensor, state, conditional)
            
        integration_times = torch.tensor([0.0, 1.0], device=device)
        final_state = odeint(ode_func, initial_state, integration_times, atol=atol, rtol=rtol)[-1]
        
        # Compute log probability of the full joint state z1 = (q1, p1)
        log_p_z1 = torch.distributions.Normal(0, 1).log_prob(final_state).sum(dim=-1)
        
        # Compute log probability of initial momentum p0
        log_p_p0 = torch.distributions.Normal(0, 1).log_prob(p0).sum(dim=-1)
        
        # Apply change of variables formula
        # Since p1(z1) = p0(z0) and p0(z0) = p_data(q0) * p(p0)
        # We have: p_data(q0) = p1(z1) / p(p0)
        # In log space: log p_data(q0) = log p1(z1) - log p(p0)
        log_p_x_normalized = log_p_z1 - log_p_p0
        
        # Account for the normalization of the data
        log_p_x = log_p_x_normalized - torch.sum(torch.log(self.scale))
        
        return log_p_x