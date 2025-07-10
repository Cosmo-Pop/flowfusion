import numpy as np
import torch
import torch.func
from torch.distributions import Normal
from torchdiffeq import odeint, odeint_adjoint
from tqdm import tqdm


class MLP(torch.nn.Module):
    """
    Multilayer perceptron for learning the score function.

    Attributes
    ----------
    n_dimensions : int
        Number of input/output dimensions
    n_conditionals : int
        Number of conditional inputs
    n_layers : int
        Number of hidden layers
    architecture : list of int
        Network architecture (input/output dims of each layer)
    activation : torch.nn.Module
        Activation function
    NN : torch.nn.ModuleList
        List of network layers
    W : torch.nn.Parameter
        Weights for the time embedding
    pi : torch.Tensor
        Tensor containing pi
    """
    def __init__(
        self,
        n_dimensions=2,
        n_conditionals=1,
        embedding_dimensions=8,
        units=[128],
        activation=torch.nn.SiLU(),
        sigma_initialization=16,
    ):
        """
        Parameters
        ----------
        n_dimensions : int, optional
            Number of input/output dimensions
        n_conditionals : int, optional
            Number of conditional inputs
        embedding_dimensions : int, optional
            Number of dimensions of the time embedding
        units : list of int, optional
            Number of hidden units per layer
        activation : torch.nn.Module, optional
            Torch activation function
        sigma_initialization : float, optional
            Standard deviation used to generate initial embedding weights
        """
        super().__init__()

        self.n_dimensions = n_dimensions
        self.n_conditionals = n_conditionals
        self.architecture = (
            [n_dimensions + n_conditionals + embedding_dimensions]
            + units
            + [n_dimensions]
        )
        self.n_layers = len(self.architecture) - 1
        self.NN = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.architecture[layer], self.architecture[layer + 1])
                for layer in range(self.n_layers)
            ]
        )
        self.W = torch.nn.Parameter(
            torch.randn(embedding_dimensions // 2) * sigma_initialization,
            requires_grad=False,
        )
        self.activation = activation

        # pi
        self.register_buffer("pi", torch.tensor(np.pi, dtype=torch.float32))

    def forward(self, t, x, conditional=None):
        """
        Forward call to the MLP

        Parameters
        ----------
        t : torch.Tensor
            Times to evaluate the score network at
        x : torch.Tensor
            Inputs to evaluate the score network at
        conditional : torch.Tensor, optional
            Conditional inputs

        Returns
        -------
        x : torch.Tensor
            Outputs from the network
        """
        # concatenate the conditional inputs
        if conditional is not None:
            x = torch.cat([x, conditional], dim=1)

        # ensure t is a tensor
        if t.size() == torch.Size([]):
            t = t * torch.ones(x.shape[:-1]).to(x.device)

        # time embedding
        t_projected = t[:, None] * self.W[None, :] * 2 * self.pi
        t_embedded = torch.cat([torch.sin(t_projected), torch.cos(t_projected)], dim=1)

        # concatenated inputs
        x = torch.cat([t_embedded, x], dim=1)

        # forward pass through layers
        for layer in range(self.n_layers - 1):
            x = self.NN[layer](x)
            x = self.activation(x)
        x = self.NN[-1](x)

        return x


class ScoreModel(torch.nn.Module):
    """
    Score-based generative model that learns the score function of 
    any given data distribution and generates samples by reversing an ODE/SDE

    Attributes
    ----------
    model : torch.nn.Module
            Score model. Usually an `MLP`.
    sde : torch.nn.Module
        Stochastic differential equation. Usually a `VPSDE`, `VESDE` or `SUBVPSDE`.
    conditional : torch.Tensor
        Internal variable for tracking conditional inputs.
    no_sigma : bool
        If `True`, `model` is assumed to return score(x, t, conditional).
        If `False`, `model` is assumed to return score(x, t, conditional) * sigma(t).
    prob : bool
        Internal variable to track whether the trace of the Jacobian is included
        in the forward call (automatically set/reset when calling `solve_odes_forward`).
    hutch : bool
        Internal variable to track whether the Skilling--Hutchinson trace estimator is 
        used in `solve_odes_forward`.
    """

    def __init__(
        self, model=None, sde=None, conditional=None, no_sigma=False, hutchinson=False
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module, optional
            Score model. Usually an `MLP`.
        sde : torch.nn.Module, optional
            Stochastic differential equation. Usually a `VPSDE`, `VESDE` or `SUBVPSDE`.
        conditional : torch.Tensor, optional
            Initial value of conditioning variable (can be updated)
        no_sigma : bool, optional
            If `True`, `model` is assumed to return score(x, t, conditional).
            If `False`, `model` is assumed to return score(x, t, conditional) * sigma(t).
        hutchinson : bool, optional
            If `True`, `solve_odes_forward` will be computed using the 
            Skilling--Hutchinson trace estimator.
        """
        super().__init__()

        # model assumed to return: score(x, t, conditional) if no_sigma is True
        # model assumed to return: score(x, t, conditional) * sigma(t) if no_sigma is False
        self.model = model  # see above
        self.sde = sde
        self.conditional = conditional  # stores the conditioning variable
        self.no_sigma = (
            no_sigma  # if True, drops the 1/sigma(t) in the score definition
        )
        self.prob = False
        self.hutch = hutchinson  # if True, uses the Hutchinson trace estimator

    def score(self, t, x, conditional=None):
        """
        Compute the time dependent score.

        Parameters
        ----------
        t : torch.Tensor
            Times to compute score at
        x : torch.Tensor
            Inputs to evaluate score at
        conditional : torch.Tensor, optional
            Conditional inputs

        Returns
        -------
        score : torch.Tensor
            Score computed via a forward pass through the score network.
        """
        if self.no_sigma:
            return self.model(t, x, conditional=conditional)

        return self.model(t, x, conditional=conditional) / self.sde.sigma(t).view(
            -1, *[1] * len(x.shape[1:])
        )

    def loss_fn(self, x, conditional=None):
        """
        Denoising score matching loss.

        Parameters
        ----------
        x : torch.Tensor
            Inputs
        conditional : torch.Tensor, optional
            Conditional inputs

        Returns
        -------
        loss : torch.Tensor
            Denoising score matching loss
        """
        return denoising_score_matching(self, x, conditional=conditional)

    def ode_drift(self, t, x, conditional=None):
        """
        Drift term in the probability flow ODE
        
        Parameters
        ----------
        t : torch.Tensor
            Times
        x : torch.Tensor
            Inputs
        conditional : torch.Tensor, optional
            Conditional inputs

        Returns
        -------
        f_tilde : torch.Tensor
            ODE drift
        """
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        f_tilde = f - 0.5 * g**2 * self.score(t, x, conditional=conditional)
        return f_tilde

    def forward(self, t, states):
        """
        Compute dx/d, and optionally dp(x)/dt at t. Input to the ODE solver.

        If `self.hutch` is `True`, the Skilling--Hutchinson trace estimator
        will be used to compute dp(x)/dt.

        Parameters
        ----------
        t : torch.Tensor
            Current times.
        states : torch.Tensor
            Current states.

        Returns
        -------
        x_dot : torch.Tensor
            Time derivative of x.
        divergence : torch.Tensor, optional
            Time derivative of p(x). Only returned if `self.prob` is `True`.

        See Also
        --------
        `sample_ode_from_base` : Integrates dx/dt to generate samples.
        `solve_odes_forward` : Integrates dx/dt and dp(x)/dt to compute log prob.
        """
        # extract x and batch size
        x = states[0]
        batchsize = x.shape[0]

        # compute derivatives of x
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)

            # Calculate the time derivative of x.
            x_dot = self.ode_drift(t, x, conditional=self.conditional)

            # Calculate the time derivative of the log determinant of the Jacobian.
            if self.prob is True:
                if self.hutch is False:
                    # Define a helper function to compute the trace of the Jacobian for a single sample.
                    def get_trace_of_jacobian(x_sample, cond_sample):
                        # Define the function whose Jacobian we want.
                        def f(x_in):
                            # Unsqueeze inputs for the model, which expects a batch dimension.
                            cond_in = cond_sample.unsqueeze(0) if cond_sample is not None else None
                            x_in_batched = x_in.unsqueeze(0)
                            
                            # Calculate the drift for the single, now-batched sample.
                            drift = self.ode_drift(t, x_in_batched, conditional=cond_in)
                            
                            # Squeeze the batch dimension from the output to match input shape.
                            return drift.squeeze(0)

                        # Compute the Jacobian of f w.r.t x_sample and return its trace.
                        return torch.trace(torch.func.jacrev(f)(x_sample))

                    #Vectorize the helper function over the batch using vmap.
                    in_dims = (0, 0) if self.conditional is not None else (0, None)
                    divergence = torch.vmap(get_trace_of_jacobian, in_dims=in_dims)(x, self.conditional)
                else:
                    divergence = torch.sum(
                        torch.autograd.grad(
                            x_dot, x, self.e, create_graph=True, retain_graph=True
                        )[0]
                        * self.e,
                        dim=1,
                    )

        if self.prob is True:
            return x_dot, divergence.view(batchsize, 1)
        else:
            return x_dot

    @torch.no_grad()
    def sample_sde(self, shape, conditional=None, steps=100):
        """
        An Euler-Maruyama integration of the model SDE backwards in time from t=T to t=0.

        Parameters
        ----------
        shape : tuple
            Shape of inputs/outputs.
        conditional : torch.Tensor, optional
            Conditional inputs.
        steps : int, optional
            Number of timesteps in SDE solution.

        Returns
        -------
        x : torch.Tensor
            Samples from the model.
        """
        batch, *dims = shape

        # prior samples
        x = (
            self.sde.prior(dims)
            .sample([batch])
            .to(next(self.model.parameters()).device)
        )

        # time step and grid
        dt = -(self.sde.T - self.sde.epsilon) / steps
        t = torch.ones(batch).to(next(self.model.parameters()).device) * self.sde.T

        # loop over SDE time steps
        for _ in (pbar := tqdm(range(steps))):
            pbar.set_description(
                f"Sampling from the prior | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                f"| scale ~ {x.max().item():.1e}"
            )
            t += dt
            if (
                t[0] < self.sde.epsilon
            ):  # Accounts for numerical error in the way we discretize t.
                break
            g = self.sde.diffusion(t, x)
            f = self.sde.drift(t, x) - g**2 * self.score(t, x, conditional=conditional)
            dw = torch.randn_like(x).to(next(self.model.parameters()).device) * (
                -dt
            ) ** (1.0 / 2.0)
            x_mean = x + f * dt
            x = x_mean + g * dw
            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        return x_mean

    # @torch.no_grad()
    def sample_ode_from_base(
        self,
        base_samples,
        conditional=None,
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
        options=None,
    ):
        """
        Generate samples deterministically by solving ODE backwards in time from t=T to t=0.

        Parameters
        ----------
        base_samples : torch.Tensor
            Base samples to transform, i.e., x(t=T) ~ N(0,1).
        conditional : torch.Tensor, optional
            Conditional inputs.
        atol : float, optional
            Absolute error tolerance for ODE solver.
        rtol : float, optional
            Relative error tolerance for ODE solver.
        method : string, optional
            ODE solving routine.
        options : dict, optional
            Dictionary of additional ODE solver options.

        Returns
        -------
        x0_samples : torch.Tensor
            Samples in parameter space generated by transforming `base_samples`.

        See Also
        --------
        `torchdiffeq.odeint` : ODE solver used (including option definitions)
        `torchdiffeq.odeint_adjoint` : ODE solver used when backward pass needed
        """

        # base samples (x(t))
        if hasattr(self.sde, "sigma_max"):
            z = base_samples * self.sde.sigma_max
        else:
            z = base_samples

        # integration times
        integration_times = torch.tensor([1.0, self.sde.epsilon]).to(z.device)

        # set prob to False
        self.prob = False

        # set the conditional if passed
        self.conditional = conditional

        # call to the ODE solver
        if self.training is True:
            state = odeint_adjoint(
                self,  # .forward() == time derivatives.
                (z,),  # state values to update.
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )
        else:
            state = odeint(
                self,  # .forward() == time derivatives.
                (z,),  # state values to update.
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )
        return state[0][1, ...], []

    @torch.no_grad()
    def solve_odes_forward(
        self,
        x0_samples,
        conditional=None,
        atol=1e-5,
        rtol=1e-5,
        method="dopri5",
        options=None,
    ):
        """
        This solves the pair of ODEs forward in time to find the base samples, x(t=T),
        and log probabilities associated with some input samples, x(t=0).

        Integrates from t=0 to t=T.

        If `self.hutch` is `True`, the Skilling--Hutchinson trace estimator will
        be used in the integrand.

        Parameters
        ----------
        x0_samples : torch.Tensor
            Samples in parameter space, i.e., x(t=0).
        conditional : torch.Tensor, optional
            Conditional inputs.
        atol : float, optional
            Absolute error tolerance for ODE solver.
        rtol : float, optional
            Relative error tolerance for ODE solver.
        method : string, optional
            ODE solving routine.
        options : dict, optional
            Dictionary of additional ODE solver options.

        Returns
        -------
        base_samples : torch.Tensor
            Samples from the base density, i.e., x(t=T).
        log_prob : torch.Tensor
            Log probability density of `x0_samples`, i.e. p[x(t=0)] - p[x(t=T)].

        See Also
        --------
        `sample_ode_from_base` : Solves in the opposite direction (from t=T to t=0).
        """

        # set prob to True
        self.prob = True

        # sample epsilons for the trace
        if self.hutch is True:
            self.e = torch.sign(torch.randn(x0_samples.shape)).to(x0_samples.device)

        # starting value of delta log px
        delta_logpx = torch.zeros(x0_samples.shape[0], 1).to(x0_samples.device)

        # integration times
        integration_times = torch.tensor([self.sde.epsilon, 1.0]).to(x0_samples.device)

        # set the conditional if passed
        self.conditional = conditional

        # call to the ODE solver
        if self.training is True:
            state = odeint_adjoint(
                self,
                (x0_samples, delta_logpx),  # state values to update.
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )
        else:
            state = odeint(
                self,
                (x0_samples, delta_logpx),  # state values to update.
                integration_times,  # when to evaluate.
                method=method,  # ode solver
                atol=atol,  # error tolerance
                rtol=rtol,  # error tolerance
                options=options,
            )

        return state[0][1, ...], state[1][1, ...]


class VESDE(torch.nn.Module):
    """
    Variance Exploding SDE.

    Attributes
    ----------
    T : torch.Tensor
        Maximum integration time of stochastic process.
    epsilon : torch.Tensor
        Minimum integration time of stochastic process.
    sigma_max : torch.Tensor
        Marginal standard deviation at t=T.
    sigma_min
        Marginal standard deviation at t=0.
    """
    def __init__(self, sigma_min=1e-2, sigma_max=10.0, T=1.0, epsilon=1e-5):
        """
        Parameters
        ----------
        sigma_min : float, optional
            Marginal standard deviation at t=0.
        sigma_max : float, optional
            Marginal standard deviation at t=T.
        T : float, optional
            Maximum integration time.
        epsilon : float, optional
            Minimum integration time.
        """
        super(VESDE, self).__init__()
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer("sigma_min", torch.tensor(sigma_min, dtype=torch.float32))
        self.register_buffer("sigma_max", torch.tensor(sigma_max, dtype=torch.float32))

    def sigma(self, t):
        """
        Marginal standard deviation at time t, sigma(t).

        Parameters
        ----------
        t : torch.Tensor
            Time to compute sigma(t) at.

        Returns
        -------
        sigma : torch.Tensor
            Marginal standard deviation, sigma(t).
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)

    def diffusion(self, t, x):
        """
        Diffusion term in the forward SDE, g(t).

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            States (not used).

        Returns
        -------
        g : torch.Tensor
            SDE diffusion, g(t).
        """
        _, *dims = x.shape  # broadcast diffusion coefficient to x shape
        return self.sigma(t).view(-1, *[1] * len(dims)) * torch.sqrt(
            2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)) / self.T
        )

    def drift(self, t, x):
        """
        Drift term in forward SDE, f(x,t)=0.

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            States.

        Returns
        -------
        f : torch.Tensor
            SDE drift, f(x,t).
        """
        return torch.zeros_like(x).to(x.device)

    def marginal_prob_scalars(self, t):
        """
        Time dependent factors in mean and standard deviation of transition kernel.
        Transition has the form p[x(t)|x(0)] = N[x(t)|nu(t)*x(0), eta^2(t)].

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        nu : torch.Tensor
            Coefficient of x(0) in mean of transition kernel.
        eta : torch.Tensor
            Standard deviation in transition kernel.
        """
        return torch.ones_like(t).to(t.device), self.sigma(t)

    def marginal_prob(self, t, x):
        """
        Mean and standard deviation of transition kernel, p[x(t)|x(0)].

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            State at t=0, i.e. x(t=0).

        Returns
        -------
        mean : torch.Tensor
            Mean of transition kernel.
        std : torch.Tensor
            Standard deviation of transition kernel.
        """

        _, *dims = x.shape

        # marginal mean and sigma
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(dims)) * x
        std = sigma_t.view(-1, *[1] * len(dims))

        return mean, std

    def sample_marginal(self, t, x0):
        """
        Sample from p[x(t)|x(0)].

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x0 : torch.Tensor
            State at t=0, i.e. x(t=0).

        Returns
        -------
        xt : torch.Tensor
            State at time t, i.e., x(t)|x(0) ~ p[x(t)|x(0)].
        """
        _, *dims = x0.shape

        # base random numbers
        z = torch.randn_like(x0).to(x0.device)

        # marginal mean and sigma
        mu_t, sigma_t = self.marginal_prob_scalars(t)

        return (
            mu_t.view(-1, *[1] * len(dims)) * x0
            + sigma_t.view(-1, *[1] * len(dims)) * z
        )

    def prior(self, shape, mu=None):
        """
        Prior distribution.

        Parameters
        ----------
        shape : tuple
            Dimensions of distribution.
        mu : torch.Tensor, optional
            Prior mean. Not recommended to set this explicitly.

        Returns
        -------
        torch.distributions.normal.Normal
            Gaussian prior.
        """
        if mu is None:
            mu = torch.zeros(shape).to(self.T.device)
        else:
            assert mu.shape == shape
        return Normal(loc=mu, scale=self.sigma_max)


class VPSDE(torch.nn.Module):
    """
    Variance Preserving SDE.

    Attributes
    ----------
    T : torch.Tensor
        Maximum integration time of stochastic process.
    epsilon : torch.Tensor
        Minimum integration time of stochastic process.
    beta_max : torch.Tensor
        Beta at t=T.
    beta_min : torch.Tensor
        Beta at t=0.
    """

    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        T=1.0,
        epsilon=1e-3,
    ):
        """
        Parameters
        ----------
        beta_min : float, optional
            Beta at t=0.
        beta_max : float, optional
            Beta at t=T.
        T : float, optional
            Maximum integration time of stochastic process.
        epsilon : float, optional
            Minimum integration time of stochastic process.
        """
        super(VPSDE, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def beta(self, t):
        """
        Computes beta(t).

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        beta : torch.Tensor
            Coefficient beta(t).
        """
        return self.beta_min + (self.beta_max - self.beta_min) * (t / self.T)

    def sigma(self, t):
        """
        Standard deviation at time t.

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        sigma : torch.Tensor
            Marginal standard deviation.
        """
        return self.marginal_prob_scalars(t)[1]

    def prior(self, shape):
        """
        Prior distribution.

        Parameters
        ----------
        shape : tuple
            Dimensions of distribution.

        Returns
        -------
        torch.distributions.normal.Normal
            Gaussian prior, N(0,1).
        """
        return Normal(loc=torch.zeros(shape).to(self.epsilon.device), scale=1.0)

    def diffusion(self, t, x):
        """
        Diffusion term in the forward SDE, g(t) = sqrt[beta(t)].

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            States (not used).

        Returns
        -------
        g : torch.Tensor
            SDE diffusion, g(t).
        """
        _, *dims = x.shape
        return torch.sqrt(self.beta(t)).view(-1, *[1] * len(dims))

    def drift(self, t, x):
        """
        Drift term in forward SDE, f(x,t)=-0.5*beta(t)*x.

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            States.

        Returns
        -------
        f : torch.Tensor
            SDE drift, f(x,t).
        """
        _, *dims = x.shape
        return -0.5 * self.beta(t).view(-1, *[1] * len(dims)) * x

    def marginal_prob_scalars(self, t):
        """
        Time dependent factors in mean and standard deviation of transition kernel.
        Transition has the form p[x(t)|x(0)] = N[x(t)|nu(t)*x(0), eta^2(t)].

        See equation (33) in Song et al (2020; arXiv:2011.13456).

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        nu : torch.Tensor
            Coefficient of x(0) in mean of transition kernel.
        eta : torch.Tensor
            Standard deviation in transition kernel.
        """
        log_coeff = (
            0.5 * (self.beta_max - self.beta_min) * t**2 / self.T + self.beta_min * t
        )  # integral of b(t)
        std = torch.sqrt(1.0 - torch.exp(-log_coeff))
        return torch.exp(-0.5 * log_coeff), std

    def marginal_prob(self, t, x):
        """
        Mean and standard deviation of transition kernel, p[x(t)|x(0)].

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            State at t=0, i.e. x(t=0).

        Returns
        -------
        mean : torch.Tensor
            Mean of transition kernel.
        std : torch.Tensor
            Standard deviation of transition kernel.
        """
        _, *dims = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(dims)) * x
        std = sigma_t.view(-1, *[1] * len(dims))
        return mean, std


class SUBVPSDE(torch.nn.Module):
    """
    Sub-Variance Preserving SDE.

    Attributes
    ----------
    T : torch.Tensor
        Maximum integration time of stochastic process.
    epsilon : torch.Tensor
        Minimum integration time of stochastic process.
    beta_max : torch.Tensor
        Beta at t=T.
    beta_min : torch.Tensor
        Beta at t=0.
    """
    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        T=1.0,
        epsilon=1e-5,
    ):
        """
        Parameters
        ----------
        beta_min : float, optional
            Beta at t=0.
        beta_max : float, optional
            Beta at t=T.
        T : float, optional
            Maximum integration time of stochastic process.
        epsilon : float, optional
            Minimum integration time of stochastic process.
        """
        super(SUBVPSDE, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def beta(self, t):
        """
        Computes beta(t).

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        beta : torch.Tensor
            Coefficient beta(t).
        """
        return self.beta_min + (self.beta_max - self.beta_min) * (t / self.T)

    def sigma(self, t):
        """
        Standard deviation at time t.

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        sigma : torch.Tensor
            Marginal standard deviation.
        """
        return self.marginal_prob_scalars(t)[1]

    def prior(self, shape):
        """
        Prior distribution.

        Parameters
        ----------
        shape : tuple
            Dimensions of distribution.

        Returns
        -------
        torch.distributions.normal.Normal
            Gaussian prior, N(0,1).
        """
        return Normal(loc=torch.zeros(shape).to(self.epsilon.device), scale=1.0)

    def diffusion(self, t, x):
        """
        Diffusion term in the forward SDE, g(t).

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            States (not used).

        Returns
        -------
        g : torch.Tensor
            SDE diffusion, g(t).
        """
        _, *dims = x.shape
        return torch.sqrt(
            self.beta(t)
            * (
                1.0
                - torch.exp(
                    self.beta_min * t
                    + 0.5 * (self.beta_max - self.beta_min) * t**2 / self.T
                )
            )
        ).view(-1, *[1] * len(dims))

    def drift(self, t, x):
        """
        Drift term in forward SDE, f(x,t)=-0.5*beta(t)*x.

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            States.

        Returns
        -------
        f : torch.Tensor
            SDE drift, f(x,t).
        """
        _, *dims = x.shape
        return -0.5 * self.beta(t).view(-1, *[1] * len(dims)) * x

    def marginal_prob_scalars(self, t):
        """
        Time dependent factors in mean and standard deviation of transition kernel.
        Transition has the form p[x(t)|x(0)] = N[x(t)|nu(t)*x(0), eta^2(t)].

        See equation (33) in Song et al (2020; arXiv:2011.13456).

        Parameters
        ----------
        t : torch.Tensor
            Times.

        Returns
        -------
        nu : torch.Tensor
            Coefficient of x(0) in mean of transition kernel.
        eta : torch.Tensor
            Standard deviation in transition kernel.
        """
        log_coeff = (
            0.5 * (self.beta_max - self.beta_min) * t**2 / self.T + self.beta_min * t
        )  # integral of b(t)
        std = 1.0 - torch.exp(-log_coeff)
        mu = torch.exp(-0.5 * log_coeff)
        return mu, std

    def marginal_prob(self, t, x):
        """
        Mean and standard deviation of transition kernel, p[x(t)|x(0)].

        Parameters
        ----------
        t : torch.Tensor
            Times.
        x : torch.Tensor
            State at t=0, i.e. x(t=0).

        Returns
        -------
        mean : torch.Tensor
            Mean of transition kernel.
        std : torch.Tensor
            Standard deviation of transition kernel.
        """
        _, *dims = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(dims)) * x
        std = sigma_t.view(-1, *[1] * len(dims))
        return mean, std


def denoising_score_matching(score_model, x, conditional=None):
    """
    Denoising score matching loss function.
    Based on Song et al. (2020; arXiv:2011.13456).

    Parameters
    ----------
    score_model : flowfusion.diffusion.ScoreModel
        Class containing the score model.
    x : torch.Tensor
        Inputs drawn from the target distribution.
    conditional : torch.Tensor, optional
        Conditional inputs.

    Returns
    -------
    loss : torch.Tensor
        Denoising score matching loss.
    """

    batch, *dims = x.shape

    # gaussian random draws
    z = torch.randn_like(x).to(x.device)

    # draw times from [\epsilon, T]
    t = (
        torch.rand(batch).to(x.device) * (score_model.sde.T - score_model.sde.epsilon)
        + score_model.sde.epsilon
    )

    # means and std-deviations of the marginal
    mean, sigma = score_model.sde.marginal_prob(t, x)

    # de-noising score matching
    return (
        torch.sum(
            (
                z
                + sigma
                * score_model.score(t, mean + sigma * z, conditional=conditional)
            )
            ** 2
        )
        / batch
    )


def log_prob_score_matching(score_model, x, conditional=None):
    """
    Score matching loss function with likelihood weighting.
    Based on Song et al. (2020; arXiv:2101.09258).

    Parameters
    ----------
    score_model : flowfusion.diffusion.ScoreModel
        Class containing the score model.
    x : torch.Tensor
        Inputs drawn from the target distribution.
    conditional : torch.Tensor, optional
        Conditional inputs.

    Returns
    -------
    loss : torch.Tensor
        Score matching loss with likelihood weighting.
    """

    batch, *dims = x.shape

    # gaussian random draws
    z = torch.randn_like(x).to(x.device)

    # draw times from [\epsilon, T]
    t = (
        torch.rand(batch).to(x.device) * (score_model.sde.T - score_model.sde.epsilon)
        + score_model.sde.epsilon
    )

    # diffusion coefficient
    g = score_model.sde.diffusion(t, x)
    # means and std-deviations of the marginal
    mean, sigma = score_model.sde.marginal_prob(t, x)

    # de-noising score matching
    return (
        torch.sum(
            (
                (g / sigma) * z
                + g * score_model.score(t, mean + sigma * z, conditional=conditional)
            )
            ** 2
        )
        / batch
    )


class PopulationModelDiffusion(torch.nn.Module):
    """
    Diffusion model class without conditionals.

    This class wraps a `ScoreModel` to provide useful functionality for
    population modelling and unconditional density estimation.

    Attributes
    ----------
    model : flowfusion.diffusion.MLP
        Score network.
    sde : torch.nn.Module
        SDE class. Usually `VESDE`, `VPSDE`, or `SUBVPSDE`.
    score_model : flowfusion.diffusion.ScoreModel
        Score model class.
    shift : torch.Tensor
        Parameter shift for inputs/outputs.
    scale : torch.Tensor
        Parameter scale for inputs/outputs.
    method : str
        Name of ODE solver.
    options : dict
        Optional arguments for ODE solver.

    See Also
    --------
    `PopulationModelDiffusionConditional` : Similar class for conditional densities.
    `ScoreModel` : Underlying class for the score-based model.
    `MLP` : Underlying class for the score network.
    """
    def __init__(
        self,
        model=None,
        sde=None,
        shift=None,
        scale=None,
        method="dopri5",
        no_sigma=False,
        hutchinson=False,
        options=None,
    ):
        """
        Parameters
        ----------
        model : flowfusion.diffusion.MLP, optional
            Score network.
        sde : torch.nn.Module, optional
            SDE class. Usually `VESDE`, `VPSDE`, or `SUBVPSDE`.
        shift : torch.Tensor, optional
            Parameter shift for inputs/outputs.
        scale : torch.Tensor, optional
            Parameter scale for inputs/outputs.
        method : str, optional
            Name of ODE solver. Must be a valid `torchdiffeq` solver name.
        no_sigma : bool, optional
            If `True`, `model` is assumed to return score(x, t, conditional).
            If `False`, `model` is assumed to return score(x, t, conditional) * sigma(t).
            For a `VPSDE`, setting `no_sigma=True` is strongly recommended.
        hutchinson : bool, optional
            If `True`, `log_prob` will be computed using the Skilling--Hutchinson 
            trace estimator.  
        options : dict, optional
            Optional arguments for ODE solver.
        """
        super(PopulationModelDiffusion, self).__init__()

        self.model = model
        self.sde = sde
        self.score_model = ScoreModel(
            model=self.model, sde=self.sde, hutchinson=hutchinson, no_sigma=no_sigma
        )
        self.register_buffer(
            "shift",
            (
                shift
                if shift is not None
                else torch.zeros(self.model.n_dimensions, dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "scale",
            (
                scale
                if scale is not None
                else torch.ones(self.model.n_dimensions, dtype=torch.float32)
            ),
        )
        self.method = method
        self.options = options

    def forward(self, base_samples):
        """
        Generate samples deterministically via an ODE solve.
        Applies any rescaling set by `self.shift` and `self.scale`.

        Parameters
        ----------
        base_samples : torch.Tensor
            Samples from base density, i.e., x(t=T) ~ N(0,1).

        Returns
        -------
        target_samples : torch.Tensor
            Rescaled samples from target density, i.e., shift + x(t=0)*scale.

        See Also
        --------
        `ScoreModel.sample_ode_from_base` : Reverse time ODE solve.
        """
        return (
            self.score_model.sample_ode_from_base(
                base_samples,
                method=self.method,
                atol=1e-5,
                rtol=1e-5,
                options=self.options,
            )[0]
            * self.scale
            + self.shift
        )

    def sample_sde(self, shape, steps=100):
        """
        Generate samples stochastically via an SDE solve.
        Applies any rescaling set by `self.shift` and `self.scale`.

        Parameters
        ----------
        shape : tuple
            Dimensions of samples to generate.
        steps : int, optional
            Number of SDE steps to take.

        Returns
        -------
        target_samples : torch.Tensor
            Rescaled samples from target density, i.e., shift + x(t=0)*scale.

        See Also
        --------
        `ScoreModel.sample_sde` : Reverse time SDE solver.
        """
        return self.score_model.sample_sde(shape, steps=100) * self.scale + self.shift

    def log_prob(self, x, atol=1e-5, rtol=1e-5):
        """
        Compute log probability of target samples, p[x(t=0)].

        Parameters
        ----------
        x : torch.Tensor
            Samples in target space, i.e., x(t=0).
        atol : float, optional
            Absolute error tolerance for ODE solver.
        rtol : float, optional
            Relative error tolerance for ODE solver.

        Returns
        -------
        lp : torch.Tensor
            Log probability density of `x`.

        See Also
        --------
        `ScoreModel.solve_odes_forward` : Forward time ODE solver.
        """
        # solve for base sample and delta log prob
        xT, lp = self.score_model.solve_odes_forward(
            (x - self.shift) / self.scale, atol=atol, rtol=rtol, options=self.options
        )

        # add log of the base density
        lp = lp + torch.sum(self.sde.prior(xT.shape).log_prob(xT), 1, keepdim=True)

        return lp


class PopulationModelDiffusionConditional(torch.nn.Module):
    """
    Diffusion model class with conditionals.

    This class wraps a `ScoreModel` to provide useful functionality for
    population modelling and conditional density estimation.

    Attributes
    ----------
    model : flowfusion.diffusion.MLP
        Score network.
    sde : torch.nn.Module
        SDE class. Usually `VESDE`, `VPSDE`, or `SUBVPSDE`.
    score_model : flowfusion.diffusion.ScoreModel
        Score model class.
    shift : torch.Tensor
        Parameter shift for inputs/outputs.
    scale : torch.Tensor
        Parameter scale for inputs/outputs.
    conditional_shift : torch.Tensor
        Parameter shift for conditional inputs.
    conditional_scale : torch.Tensor
        Parameter scale for conditional inputs.
    method : str
        Name of ODE solver.
    options : dict
        Optional arguments for ODE solver.

    See Also
    --------
    `PopulationModelDiffusion` : Similar class for unconditional densities.
    `ScoreModel` : Underlying class for the score-based model.
    `MLP` : Underlying class for the score network.
    """

    def __init__(
        self,
        model=None,
        sde=None,
        shift=None,
        scale=None,
        conditional_shift=None,
        conditional_scale=None,
        no_sigma=False,
        method="dopri5",
        options=None,
    ):
        """
        Parameters
        ----------
        model : flowfusion.diffusion.MLP, optional
            Score network.
        sde : torch.nn.Module, optional
            SDE class. Usually `VESDE`, `VPSDE`, or `SUBVPSDE`.
        shift : torch.Tensor, optional
            Parameter shift for inputs/outputs.
        scale : torch.Tensor, optional
            Parameter scale for inputs/outputs.
        conditional_shift : torch.Tensor, optional
            Parameter shift for conditional inputs.
        conditional_scale : torch.Tensor, optional
            Parameter scale for conditional inputs.
        no_sigma : bool, optional
            If `True`, `model` is assumed to return score(x, t, conditional).
            If `False`, `model` is assumed to return score(x, t, conditional) * sigma(t).
        method : str, optional
            Name of ODE solver. Must be a valid `torchdiffeq` solver name.
        options : dict, optional
            Optional arguments for ODE solver.
        """

        super(PopulationModelDiffusionConditional, self).__init__()

        self.model = model
        self.sde = sde
        self.score_model = ScoreModel(model=self.model, sde=self.sde, no_sigma=no_sigma)
        self.register_buffer(
            "shift",
            (
                shift
                if shift is not None
                else torch.zeros(self.model.n_dimensions, dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "scale",
            (
                scale
                if scale is not None
                else torch.ones(self.model.n_dimensions, dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "conditional_shift",
            (
                conditional_shift
                if conditional_shift is not None
                else torch.zeros(self.model.n_conditionals, dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "conditional_scale",
            (
                conditional_scale
                if conditional_scale is not None
                else torch.ones(self.model.n_conditionals, dtype=torch.float32)
            ),
        )
        self.options = options
        self.method = method

    def forward(self, base_samples, conditional=None):
        """
        Generate samples deterministically via an ODE solve.
        Applies any rescaling set by `self.shift` and `self.scale`.

        Parameters
        ----------
        base_samples : torch.Tensor
            Samples from base density, i.e., x(t=T) ~ N(0,1).
        conditional : torch.Tensor, optional
            Conditional inputs.

        Returns
        -------
        target_samples : torch.Tensor
            Rescaled samples from target density given `conditional`.
        """

        return (
            self.score_model.sample_ode_from_base(
                base_samples,
                conditional=(conditional - self.conditional_shift)
                / self.conditional_scale,
                method=self.method,
                atol=1e-5,
                rtol=1e-5,
                options=self.options,
            )[0]
            * self.scale
            + self.shift
        )

    def sample_sde(self, shape, conditional=None, steps=100):
        """
        Generate samples stochastically via an SDE solve.
        Applies any rescaling set by `self.shift` and `self.scale`.

        Parameters
        ----------
        shape : tuple
            Dimensions of samples to generate.
        conditional : torch.Tensor, optional
            Conditional inputs.
        steps : int, optional
            Number of SDE steps to take.

        Returns
        -------
        target_samples : torch.Tensor
            Rescaled samples from target density given `conditional`.
        """
        return (
            self.score_model.sample_sde(
                shape,
                conditional=(conditional - self.conditional_shift)
                / self.conditional_scale,
                steps=100,
            )
            * self.scale
            + self.shift
        )

    def log_prob(self, x, conditional=None, atol=1e-5, rtol=1e-5):
        """
        Compute conditional log probability of target samples, p[x(t=0)|cond].

        Parameters
        ----------
        x : torch.Tensor
            Samples in target space.
        conditional : torch.Tensor, optional
            Conditionals corresponding to `x`.
        atol : float, optional
            Absolute error tolerance for ODE solver.
        rtol : float, optional
            Relative error tolerance for ODE solver.

        Returns
        -------
        lp : torch.Tensor
            Log probability density of `x` given `conditional`.
        """
        # solve for base sample and delta log prob
        xT, lp = self.score_model.solve_odes_forward(
            (x - self.shift) / self.scale,
            conditional=(conditional - self.conditional_shift) / self.conditional_scale,
            atol=atol,
            rtol=rtol,
            options=self.options,
        )

        # add log of the base density
        lp = lp + torch.sum(self.sde.prior(xT.shape).log_prob(xT), 1, keepdim=True)

        return lp