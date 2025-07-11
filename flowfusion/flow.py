import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import torch.nn.functional as F
from typing import Optional
from typing import Tuple, List


class ODEFlow(nn.Module):
    """
    ODE Flow model.
    
    This class implements a conditional ODE flow using a neural network to model
    the dynamics. It includes methods for computing the dynamics, sampling, 
    and calculating the flow matching loss.

    Note that we define the ODE transform from a base unit normal (at t=T) to a target
    distribution (at t=0), i.e., the forward transform integrates backwards in time,
    consistent with diffusion models.

    Attributes
    ----------
    target_dimension : int
        Dimension of the target distribution.
    layers : torch.nn.ModuleList
        List of neural network layers.
    velocity : torch.nn.Sequential
        Model for the dynamics of the flow.
    target_shift : torch.Tensor
        Shift to be applied to the target distribution.
    target_scale : torch.Tensor
        Scale to be applied to the target distribution.
    twopi : torch.Tensor
        Tensor containing 2*pi.
    """

    def __init__(
        self,
        target_dimension: int = 1,
        hidden_units: List[int] = [128, 128],
        activation: nn.Module = nn.SiLU,
        target_shift: Optional[torch.Tensor] = None,
        target_scale: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        target_dimension : int, optional
            Dimension of the target distribution.
        hidden_units : list of int, optional
            Number of hidden units per layer in the dynamics network.
        activation : torch.nn.Module, optional
            Activation function.
        target_shift : torch.Tensor, optional
            Shift to be applied to the target distribution.
        target_scale : torch.Tensor, optional
            Scale to be applied to the target distribution.
        """
        super().__init__()

        # make a module list of layers
        self.target_dimension = target_dimension
        self.layers = nn.ModuleList()
        architecture = (
            [target_dimension + 1] + hidden_units + [target_dimension]
        )  # +1 for time
        n_layers = len(architecture) - 1
        for layer in range(n_layers - 1):
            self.layers.append(nn.Linear(architecture[layer], architecture[layer + 1]))
            self.layers.append(activation())
        self.layers.append(nn.Linear(architecture[-2], architecture[-1]))

        # make a sqeuential model for the dynamics
        self.velocity = nn.Sequential(*self.layers)

        # two pi
        self.register_buffer("twopi", torch.tensor(2.0 * 3.14159265358979323846))

        # shift and scale
        self.register_buffer(
            "target_shift",
            target_shift if target_shift is not None else torch.zeros(target_dimension),
        )
        self.register_buffer(
            "target_scale",
            target_scale if target_scale is not None else torch.ones(target_dimension),
        )

    def dynamics(self, t: torch.Tensor, states: Tuple[torch.Tensor]):
        """
        Computes the dynamics, dx/dt, of the ODE flow.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        states : tuple of torch.Tensor
            Tuple containing the current state.
            Current state (x) is assumed to be in `states[0]`. 
            Note that x here is assumed to be normalized by the shift and scale.
        
        Returns
        -------
        dxdt : torch.Tensor
            The velocity field.
        """

        # extract states
        x = states[0]

        # assemble the inputs
        inputs = torch.cat(
            [x, t.view(-1, 1).expand(x.shape[0], 1)],
            dim=1,  # make sure t has the right shape
        )

        # compute the velocity field
        dxdt = self.velocity(inputs)

        return dxdt

    def dynamics_with_jacobian(
        self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Computes the dynamics of the ODE flow and the log determinant of the Jacobian.
        These correspond to dx/dt and dp(x)/dt.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        states : tuple of torch.Tensor
            Tuple containing the current state, and log Jacobian.
            Current state (x) is assumed to be in `states[0]`. 
            Log Jacobian assumed to be in `states[1]`.
            Note that x here is assumed to be normalized by the shift and scale.
        Returns
        -------
        dxdt : torch.Tensor
            The velocity field.
        divergence : torch.Tensor
            Divergence of the velocity field.
        """
        # pull out the x, jacobian
        x, log_jacobian = states

        # compute derivatives of x
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)

            # Calculate the time derivative of x.
            dxdt = self.dynamics(t, (x,))  # (batch_size, n_covariates, seq_length)

            # Calculate the time derivative of the log determinant of the Jacobian.
            divergence = torch.zeros_like(log_jacobian)
            for i in range(0, x.shape[-1]):
                divergence = divergence + torch.autograd.grad(
                    dxdt[:, i].sum(), x, create_graph=True, retain_graph=True
                )[0][:, i].unsqueeze(1)

        return (
            dxdt,  # velocity field
            divergence,  # divergence of the velocity field
        )

    # for the ODE solver: velocity field
    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor]):
        """
        Computes the forward pass of the ODE flow, i.e., the velocity field.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        states : tuple of torch.Tensor
            Tuple containing the current state.
            Current state (x) is assumed to be in `states[0]`. 
            Note that x here is assumed to be normalized by the shift and scale.
        
        Returns
        -------
        dxdt : torch.Tensor
            The velocity field.
        """

        return self.dynamics(t, states)

    # function to compute the ideal linear velocity field
    def compute_linear_velocity_field(
        self,
        x0: torch.Tensor,  # initial state (observed samples)
        xT: torch.Tensor,  # final state (base density)
        t: torch.Tensor,  # time
    ):
        """
        Computes the ideal linear velocity field for the ODE flow.

        Parameters
        ----------
        x0 : torch.Tensor
            Initial state at t=0 (observed samples).
        xT : torch.Tensor
            Final state at t=T (base density).
        t : torch.Tensor
            Time tensor.
        Returns
        -------
        xt : torch.Tensor
            Interpolated state at times t.
        v_hat : torch.Tensor
            Ideal velocity (dx/dt).
        """

        # shift and scale x0: note that the model learns the dynamics for the normalized variable (via target shift and scale)
        x0 = (x0 - self.target_shift) / self.target_scale

        xt = (1 - t) * x0 + t * xT  # linear interpolation: t=1 x=xT, t=0 x=x0
        v_hat = xT - x0  # ideal velocity (dx/dt)
        return xt, v_hat

    def flow_matching_loss(
        self,
        x: torch.Tensor,  # initial state (observed samples)
    ):
        """
        Computes the flow matching loss for the ODE flow.

        Parameters
        ----------
        x : torch.Tensor
            Initial state at t=0 (observed samples).
        
        Returns
        -------
        loss : torch.Tensor
            Flow matching loss.
        """

        # base samples
        xT = torch.randn_like(x)

        # sample random t in [0,1]
        t = torch.rand(x.shape[0], 1).to(x.device)

        # compute the interpolated position and ideal velocity
        xt, v_hat = self.compute_linear_velocity_field(x, xT, t)

        # predict velocity using the neural network
        v_pred = self.dynamics(t, (xt,))

        # compute L2 loss between predicted and ideal velocity
        loss = torch.mean((v_pred - v_hat) ** 2)

        return loss

    # function to transform base samples to the target under the ODE flow via odeint
    def sample(
        self,
        xT: torch.Tensor,
        gradients: bool = False,
    ):
        """
        This function transforms base samples to the target under the ODE flow
        via odeint, by integrating backwards in time from t=T to t=0.

        Parameters
        ----------
        xT : torch.Tensor
            Final state at t=T (base density).
        gradients : bool, optional
            Whether to compute in a `torch.no_grad()` context.

        Returns
        -------
        x0 : torch.Tensor 
            Transformed samples from target density at t=0.
        """

        # integration times (note: solving backwards in time)
        integration_times = torch.tensor([1.0, 0.0]).to(xT.device)

        # solve ODE (backwards in time) and return final state (t=0 -> target samples)
        # note we need to rescale the outputs using target shift and scale
        if gradients:
            return (
                odeint_adjoint(
                    self,
                    (xT,),  # initial state
                    integration_times,  # evaluation times (backwards in time)
                )[0][-1]
                * self.target_scale
                + self.target_shift
            )
        else:
            with torch.no_grad():
                return (
                    odeint(
                        self.dynamics,
                        (xT,),  # initial state
                        integration_times,  # evaluation times (backwards in time)
                    )[0][-1]
                    * self.target_scale
                    + self.target_shift
                )

    def solve_ode_forward(
        self,
        x: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        method: str = "dopri5",
        options: Optional[dict] = None,
        adjoint: bool = False,
    ):
        """
        This solves the pair of ODEs forward in time (t=0 (target) -> t=T (base))
        to find the base x(t=T) samples and log probabilities associated with 
        some input x(t=0) samples.

        Parameters
        ----------
        x : torch.Tensor
            The initial state x(t=0) (observed samples). 
            Note this is assumed to be normalized by the shift and scale.
            Shape: (batch_size, n_covariates)
        atol : float, optional
            Absolute error tolerance for the ODE solver.
        rtol : float, optional
            Relative error tolerance for the ODE solver.
        method : str, optional
            The ODE solver method to use. Must be a valid `torchdiffeq` option.
        options : dict, optional
            Additional options for the ODE solver.
        adjoint : bool, optional
            Whether to use the adjoint ODE solver.

        Returns
        -------
        xT : torch.Tensor
            State x(t=T), i.e., samples from the base density.
        lp : torch.Tensor
            Log probabilities of input samples x(t=0).
        """

        # pull out the shapes
        batch_size, _ = x.size()

        # starting value of delta log px
        log_jacobian = torch.zeros(batch_size, 1).to(x.device)

        # integration times
        integration_times = torch.tensor([0.0, 1.0]).to(x.device)

        # call to the ODE solver
        if adjoint:
            state = odeint_adjoint(
                self.dynamics_with_jacobian,
                (
                    x,  # initial state (observations)
                    log_jacobian,  # initialized log jacobian
                ),  # state
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )
        else:
            state = odeint(
                self.dynamics_with_jacobian,
                (
                    x,  # initial state (observations)
                    log_jacobian,  # initialized log jacobian
                ),  # state
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )

        return state[0][1, ...], state[1][1, ...]  # samples x(t=T), log jacobian

    def log_prob(
        self,
        x: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        method: str = "dopri5",
        options: Optional[dict] = None,
        adjoint: bool = False,
    ):
        """
        Computes the log probability of the input samples under the ODE flow.
        This includes the correction for the probabilities under the base density.

        Parameters
        ----------
        x : torch.Tensor
            Input samples, x(t=0).
        atol : float, optional
            Absolute tolerance for the ODE solver.
        rtol : float, optional
            Relative tolerance for the ODE solver.
        method : str, optional
            ODE solver method. Must be a valid `torchdiffeq` option.
        options : dict, optional
            Additional options for the ODE solver.
        adjoint : bool, optional
            Whether to use the adjoint ODE solver.

        Returns
        -------
        log_prob : torch.Tensor
            Log probability of the input samples.
        """

        # shift and scale the target
        x = (x - self.target_shift) / self.target_scale

        # solve the ODE forward in time (0 (target) -> T (base))
        xT, log_jacobian = self.solve_ode_forward(
            x,
            atol,
            rtol,
            method,
            options,
            adjoint,
        )

        # prior log-probability
        log_prob = torch.sum(-0.5 * xT**2 - 0.5 * torch.log(self.twopi), dim=1)

        return (
            log_prob + log_jacobian.squeeze(1) - torch.sum(torch.log(self.target_scale))
        )


class ConditionalODEFlow(nn.Module):
    """
    This class implements a conditional ODE flow using a neural network to model
    the dynamics. It includes methods for computing the dynamics, sampling, 
    and calculating the flow matching loss.

    Note that we define the ODE transform from a base unit normal (at t=T) to a target
    distribution (at t=0), i.e., the forward transform integrates backwards in time,
    consistent with diffusion models.

    Attributes
    ----------
    target_dimension : int
        Dimension of the target distribution.
    conditional_dimension : int
        Dimension of the conditional inputs.
    layers : torch.nn.ModuleList
        List of neural network layers.
    velocity : torch.nn.Sequential
        Model for the dynamics of the flow.
    target_shift : torch.Tensor
        Shift to be applied to the target distribution.
    target_scale : torch.Tensor
        Scale to be applied to the target distribution.
    conditional_shift : torch.Tensor
        Shift to be applied to the conditional inputs.
    conditional_scale : torch.Tensor
        Scale to be applied to the conditional inputs.
    twopi : torch.Tensor
        Tensor containing 2*pi.
    """

    def __init__(
        self,
        target_dimension: int = 1,
        conditional_dimension: int = 1,
        hidden_units: List[int] = [128, 128],
        activation: nn.Module = nn.SiLU,
        target_shift: Optional[torch.Tensor] = None,
        target_scale: Optional[torch.Tensor] = None,
        conditional_shift: Optional[torch.Tensor] = None,
        conditional_scale: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        target_dimension : int, optional
            Dimension of the target distribution.
        conditional_dimension : int, optional
            Dimension of the conditional inputs.
        hidden_units : list of int, optional
            Number of hidden units per layer in the dynamics network.
        activation : torch.nn.Module, optional
            Activation function.
        target_shift : torch.Tensor, optional
            Shift to be applied to the target distribution.
        target_scale : torch.Tensor, optional
            Scale to be applied to the target distribution.
        conditional_shift : torch.Tensor, optional
            Shift to be applied to the conditional inputs.
        conditional_scale : torch.Tensor, optional
            Scale to be applied to the conditional inputs.
        """
        super().__init__()

        # make a module list of layers
        self.target_dimension = target_dimension
        self.conditional_dimension = conditional_dimension
        self.layers = nn.ModuleList()
        architecture = (
            [target_dimension + 1 + conditional_dimension]
            + hidden_units
            + [target_dimension]
        )  # +1 for time
        n_layers = len(architecture) - 1
        for layer in range(n_layers - 1):
            self.layers.append(nn.Linear(architecture[layer], architecture[layer + 1]))
            self.layers.append(activation())
        self.layers.append(nn.Linear(architecture[-2], architecture[-1]))

        # make a sqeuential model for the dynamics
        self.velocity = nn.Sequential(*self.layers)

        # two pi
        self.register_buffer("twopi", torch.tensor(2.0 * 3.14159265358979323846))

        # shift and scale (register buffers)
        self.register_buffer(
            "target_shift",
            target_shift if target_shift is not None else torch.zeros(target_dimension),
        )
        self.register_buffer(
            "target_scale",
            target_scale if target_scale is not None else torch.ones(target_dimension),
        )
        self.register_buffer(
            "conditional_shift",
            (
                conditional_shift
                if conditional_shift is not None
                else torch.zeros(conditional_dimension)
            ),
        )
        self.register_buffer(
            "conditional_scale",
            (
                conditional_scale
                if conditional_scale is not None
                else torch.ones(conditional_dimension)
            ),
        )

    def dynamics(self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        """
        Computes the dynamics, dx/dt,of the ODE flow.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        states : tuple of torch.Tensor
            Tuple containing the current state and conditional inputs.
            Current state (x) is assumed to be in `states[0]`. 
            Conditional inputs are assumed to be in `states[1]`.
            Note that x here is assumed to be normalized by the shift and scale.
            Note that the conditionals get normalized inside the function.

        Returns
        -------
        dxdt : torch.Tensor
            The velocity field.
        zeros : torch.Tensor
            Zero gradient for the conditionals.
        """

        # extract states
        x, conditional = states

        # shift and scale the conditionals
        conditional = (conditional - self.conditional_shift) / self.conditional_scale

        # assemble the inputs
        inputs = torch.cat(
            [x, t.view(-1, 1).expand(x.shape[0], 1), conditional],
            dim=1,  # make sure t has the right shape
        )

        # compute the velocity field
        dxdt = self.velocity(inputs)

        return (
            dxdt,  # velocity field
            torch.zeros_like(conditional).to(
                x.device
            ),  # zero gradient for the conditionals
        )

    def dynamics_with_jacobian(
        self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Computes the dynamics, dx/dt, of the ODE flow
        and the log determinant of the Jacobian, dp(x)/dt.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        states : tuple of torch.Tensor
            Tuple containing the current state, conditional inputs, and log Jacobian.
            Current state (x) is assumed to be in `states[0]`. 
            Conditional inputs are assumed to be in `states[1]`.
            Log Jacobian assumed to be in `states[2]`.
            Note that x here is assumed to be normalized by the shift and scale.
            Note that the conditionals get normalized inside the function.
        
        Returns
        -------
        dxdt : torch.Tensor
            The velocity field.
        zeros : torch.Tensor
            Zero gradient for the conditionals.
        divergence : torch.Tensor
            Divergence of the velocity field.
        """
        # pull out the x, conditional, jacobian
        x, conditional, log_jacobian = states

        # compute derivatives of x
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)

            # Calculate the time derivative of x.
            dxdt = self.dynamics(t, (x, conditional))[
                0
            ]  # (batch_size, n_covariates, seq_length)

            # Calculate the time derivative of the log determinant of the Jacobian.
            divergence = torch.zeros_like(log_jacobian)
            for i in range(0, x.shape[-1]):
                divergence = divergence + torch.autograd.grad(
                    dxdt[:, i].sum(), x, create_graph=True, retain_graph=True
                )[0][:, i].unsqueeze(1)

        return (
            dxdt,  # velocity field
            torch.zeros_like(
                conditional, device=x.device
            ),  # conditional gradients (zero)
            divergence,  # divergence of the velocity field
        )

    # for the ODE solver: velocity field
    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        """
        Computes the forward pass of the conditional ODE flow.
        
        Parameters
        ----------
        t : torch.Tensor
            Time tensor.
        states : tuple of torch.Tensor
            Tuple containing the current state and conditional inputs.
            Current state (x) is assumed to be in `states[0]`. 
            Conditional inputs are assumed to be in `states[1]`.
            Note that x here is assumed to be normalized by the shift and scale.
            Note that the conditionals get normalized inside the function.
        
        Returns
        -------
        dxdt : torch.Tensor
            The velocity field.
        """

        return self.dynamics(t, states)

    # function to compute the ideal linear velocity field
    def compute_linear_velocity_field(
        self,
        x0: torch.Tensor,  # initial state (observed samples)
        xT: torch.Tensor,  # final state (base density)
        t: torch.Tensor,  # time
    ):
        """
        Computes the ideal linear velocity field for the ODE flow.

        Parameters
        ----------
        x0 : torch.Tensor
            Initial state at t=0 (observed samples).
        xT : torch.Tensor
            Final state at t=T (base density).
        t : torch.Tensor
            Time tensor.
        Returns
        -------
        xt : torch.Tensor
            Interpolated state at times t.
        v_hat : torch.Tensor
            Ideal velocity (dx/dt).
        """

        # shift the target samples. Note that the model learns the dynamics for the normalized variable (via target shift and scale)
        x0 = (x0 - self.target_shift) / self.target_scale

        xt = (1 - t) * x0 + t * xT  # linear interpolation: t=1 x=xT, t=0 x=x0
        v_hat = xT - x0  # ideal velocity (dx/dt)
        return xt, v_hat

    def flow_matching_loss(
        self,
        x: torch.Tensor,
        conditional: torch.Tensor,
    ):
        """
        Computes the flow matching loss for the ODE flow.

        Parameters
        ----------
        x : torch.Tensor
            Initial state at t=0 (observed samples).
        conditional : torch.Tensor
            Conditional inputs.
        
        Returns
        -------
        loss : torch.Tensor
            Flow matching loss.
        """

        # base samples
        xT = torch.randn_like(x)

        # sample random t in [0,1]
        t = torch.rand(x.shape[0], 1).to(x.device)

        # compute the interpolated position and ideal velocity
        xt, v_hat = self.compute_linear_velocity_field(x, xT, t)

        # predict velocity using the neural network
        v_pred = self.dynamics(t, (xt, conditional))[0]

        # compute L2 loss between predicted and ideal velocity
        loss = torch.mean((v_pred - v_hat) ** 2)

        return loss

    # function to transform base samples to the target under the ODE flow via odeint
    def sample(
        self,
        xT: torch.Tensor,  # final state (base density)
        conditional: torch.Tensor,  # conditional inputs
        gradients: bool = False,
    ):
        """
        This function transforms base samples to the target under the ODE flow
        via odeint, by integrating backwards in time from t=T to t=0.

        Parameters
        ----------
        xT : torch.Tensor
            Final state at t=T (base density).
        conditional : torch.Tensor
            Conditional inputs.
        gradients : bool, optional
            Whether to compute in a `torch.no_grad()` context.

        Returns
        -------
        x0 : torch.Tensor 
            Transformed samples from target density at t=0.
        """
        # integration times (note: solving backwards in time)
        integration_times = torch.tensor([1.0, 0.0]).to(xT.device)

        # solve ODE (backwards in time) and return final state (t=0 -> target samples)
        # note we need to rescale the outputs using target shift and scale
        if gradients:
            return (
                odeint_adjoint(
                    self,
                    (xT, conditional),  # initial state
                    integration_times,  # evaluation times
                )[0][-1]
                * self.target_scale
                + self.target_shift
            )
        else:
            with torch.no_grad():
                return (
                    odeint(
                        self.dynamics,
                        (xT, conditional),  # initial state
                        integration_times,  # evaluation times (backwards in time)
                    )[0][-1]
                    * self.target_scale
                    + self.target_shift
                )

    def solve_ode_forward(
        self,
        x: torch.Tensor,
        conditional: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        method: str = "dopri5",
        options: Optional[dict] = None,
        adjoint: bool = False,
    ):
        """
        This solves the pair of ODEs forward in time (t=0 (target) -> t=T (base))
        to find the base x(t=T) samples and log probabilities associated with 
        some input x(t=0) samples given the conditional inputs.

        Parameters
        ----------
        x : torch.Tensor
            The initial state x(t=0) (observed samples). 
            Note this is assumed to be normalized by the shift and scale.
            Shape: (batch_size, n_covariates)
        conditional : torch.Tensor
            Conditional inputs.
            Shape: (batch_size, n_conditionals)
        atol : float, optional
            Absolute error tolerance for the ODE solver.
        rtol : float, optional
            Relative error tolerance for the ODE solver.
        method : str, optional
            The ODE solver method to use. Must be a valid `torchdiffeq` option.
        options : dict, optional
            Additional options for the ODE solver.
        adjoint : bool, optional
            Whether to use the adjoint ODE solver.

        Returns
        -------
        xT : torch.Tensor
            State x(t=T), i.e., samples from the base density.
        lp : torch.Tensor
            Log probabilities of input samples x(t=0).
        """

        # pull out the shapes
        batch_size, _ = x.size()

        # starting value of delta log px
        log_jacobian = torch.zeros(batch_size, 1).to(x.device)

        # integration times
        integration_times = torch.tensor([0.0, 1.0]).to(x.device)

        # call to the ODE solver
        if adjoint:
            state = odeint_adjoint(
                self.dynamics_with_jacobian,
                (
                    x,  # initial state (observations)
                    conditional,  # conditional inputs
                    log_jacobian,  # initialized log jacobian
                ),  # state
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )
        else:
            state = odeint(
                self.dynamics_with_jacobian,
                (
                    x,  # initial state (observations)
                    conditional,  # conditional inputs
                    log_jacobian,  # initialized log jacobian
                ),  # state
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )

        return state[0][1, ...], state[-1][1, ...]  # samples y_T, log jacobian

    def log_prob(
        self,
        x: torch.Tensor,
        conditional: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        method: str = "dopri5",
        options: Optional[dict] = None,
        adjoint: bool = False,
    ):
        """
        Computes the conditional log probability of the input samples under the ODE flow.
        This includes the correction for the probabilities under the base density.

        Parameters
        ----------
        x : torch.Tensor
            Input samples, x(t=0).
        conditional : torch.Tensor
            Conditional inputs.
        atol : float, optional
            Absolute tolerance for the ODE solver.
        rtol : float, optional
            Relative tolerance for the ODE solver.
        method : str, optional
            ODE solver method. Must be a valid `torchdiffeq` option.
        options : dict, optional
            Additional options for the ODE solver.
        adjoint : bool, optional
            Whether to use the adjoint ODE solver.

        Returns
        -------
        log_prob : torch.Tensor
            Log probability of the input samples.
        """

        # shift the inputs
        x = (x - self.target_shift) / self.target_scale

        # solve the ODE forward in time to compute log prob (0 (target) -> T (base))
        xT, log_jacobian = self.solve_ode_forward(
            x,
            conditional,
            atol,
            rtol,
            method,
            options,
            adjoint,
        )

        # prior log-probability
        log_prob = torch.sum(-0.5 * xT**2 - 0.5 * torch.log(self.twopi), dim=1)

        return (
            log_prob + log_jacobian.squeeze(1) - torch.sum(torch.log(self.target_scale))
        )
