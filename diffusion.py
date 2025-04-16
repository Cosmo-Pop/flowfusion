import numpy as np
import torch
from torch.distributions import Normal
from torchdiffeq import odeint, odeint_adjoint
from tqdm import tqdm


class MLP(torch.nn.Module):
    """
    Multilayer perceptron for learning the score function
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
        """
        # concatenate the conditional inputs
        if conditional is not None:
            x = torch.cat([x, conditional], dim=1)

        # ensure t is a tensor
        if t.size() == torch.Size([]):
            t = t * torch.ones(x.shape[:-1]).to(x.device)

        # time ebmedding
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
    Score-based generative model that learns the score function of any given data distribution and generates samples by reversing an ODE/SDE
    """

    def __init__(
        self, model=None, sde=None, conditional=None, hutchinson=False
    ):
        super().__init__()

        self.model = model  # see above
        self.sde = sde
        self.conditional = conditional  # stores the conditioning variable
        self.prob = False
        self.hutch = hutchinson  # if True, uses the Hutchinson trace estimator

    def score(self, t, x, conditional=None):
            return self.model(t, x, conditional=conditional)


    def loss_fn(self, x, conditional=None):
        return denoising_score_matching(self, x, conditional=conditional)

    def ode_drift(self, t, x, conditional=None):
        f = self.sde.drift(t, x)
        g = self.sde.diffusion(t, x)
        f_tilde = f - 0.5 * g**2 * self.score(t, x, conditional=conditional)
        return f_tilde

    # define the forward pass as the ode dx/dt, for torchdiffeq's benefit later (also include dP(x)/dt if self.prob=True, and will condition on self.conditional if this is not None)
    def forward(self, t, states):

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
                    divergence = torch.autograd.grad(
                        x_dot[:, 0].sum(), x, create_graph=True, retain_graph=True
                    )[0][:, 0]
                    for i in range(1, x.shape[1]):
                        divergence = (
                            divergence
                            + torch.autograd.grad(
                                x_dot[:, i].sum(),
                                x,
                                create_graph=True,
                                retain_graph=True,
                            )[0][:, i]
                        )
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
        An Euler-Maruyama integration of the model SDE

        shape: Shape of the tensor to sample (including batch size)
        steps: Number of Euler-Maruyam steps to perform
        likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
        guidance_factor: Multiplicative factor for the likelihood drift
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
        This solves the pair of ODEs forward in time to find the base x(t=T) samples and log probabilities associated with some input x(t=0) samples
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
    def __init__(self, sigma_min=1e-2, sigma_max=10.0, T=1.0, epsilon=1e-5):
        """
        Variance Exploding stochastic differential equation

        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
        """
        super(VESDE, self).__init__()
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer("sigma_min", torch.tensor(sigma_min, dtype=torch.float32))
        self.register_buffer("sigma_max", torch.tensor(sigma_max, dtype=torch.float32))

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)

    def diffusion(self, t, x):
        _, *dims = x.shape  # broadcast diffusion coefficient to x shape
        return self.sigma(t).view(-1, *[1] * len(dims)) * torch.sqrt(
            2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)) / self.T
        )

    def drift(self, t, x):
        return torch.zeros_like(x).to(x.device)

    def marginal_prob_scalars(self, t):
        return torch.ones_like(t).to(t.device), self.sigma(t)

    def marginal_prob(self, t, x):

        _, *dims = x.shape

        # marginal mean and sigma
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(dims)) * x
        std = sigma_t.view(-1, *[1] * len(dims))

        return mean, std

    def sample_marginal(self, t, x0):
        """
        Sample from the marginal at time t given some initial condition x0
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
        Technically, VESDE does not change the mean of the 0 temperature distribution,
        so I give the option to provide for more accuracy. In practice,
        sigma_max is chosen large enough to make this choice irrelevant
        """
        if mu is None:
            mu = torch.zeros(shape).to(self.T.device)
        else:
            assert mu.shape == shape
        return Normal(loc=mu, scale=self.sigma_max)


class VPSDE(torch.nn.Module):
    """
    Variance preserving stochastic differential equation.
    """

    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        T=1.0,
        epsilon=1e-3,
    ):
        super(VPSDE, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * (t / self.T)

    def sigma(self, t):
        return self.marginal_prob_scalars(t)[1]

    def prior(self, shape):
        return Normal(loc=torch.zeros(shape).to(self.epsilon.device), scale=1.0)

    def diffusion(self, t, x):
        _, *dims = x.shape
        return torch.sqrt(self.beta(t)).view(-1, *[1] * len(dims))

    def drift(self, t, x):
        _, *dims = x.shape
        return -0.5 * self.beta(t).view(-1, *[1] * len(dims)) * x

    def marginal_prob_scalars(self, t):
        """
        See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456)
        """
        log_coeff = (
            0.5 * (self.beta_max - self.beta_min) * t**2 / self.T + self.beta_min * t
        )  # integral of b(t)
        std = torch.sqrt(1.0 - torch.exp(-log_coeff))
        return torch.exp(-0.5 * log_coeff), std

    def marginal_prob(self, t, x):
        _, *dims = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(dims)) * x
        std = sigma_t.view(-1, *[1] * len(dims))
        return mean, std


class SUBVPSDE(torch.nn.Module):
    """
    Sub-variance preserving stochastic differential equation.
    """

    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        T=1.0,
        epsilon=1e-5,
    ):
        super(SUBVPSDE, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * (t / self.T)

    def sigma(self, t):
        return self.marginal_prob_scalars(t)[1]

    def prior(self, shape):
        return Normal(loc=torch.zeros(shape).to(self.epsilon.device), scale=1.0)

    def diffusion(self, t, x):
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
        _, *dims = x.shape
        return -0.5 * self.beta(t).view(-1, *[1] * len(dims)) * x

    def marginal_prob_scalars(self, t):
        """
        See equation (33) in Song et al 2020. (https://arxiv.org/abs/2011.13456)
        """
        log_coeff = (
            0.5 * (self.beta_max - self.beta_min) * t**2 / self.T + self.beta_min * t
        )  # integral of b(t)
        std = 1.0 - torch.exp(-log_coeff)
        mu = torch.exp(-0.5 * log_coeff)
        return mu, std

    def marginal_prob(self, t, x):
        _, *dims = x.shape
        m_t, sigma_t = self.marginal_prob_scalars(t)
        mean = m_t.view(-1, *[1] * len(dims)) * x
        std = sigma_t.view(-1, *[1] * len(dims))
        return mean, std


def denoising_score_matching(score_model, x, conditional=None):
    """
    Training function for the diffusion model with denoising score matching
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
    Training function for the diffusion model with log-prob
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
    def __init__(
        self,
        model=None,
        sde=None,
        shift=None,
        scale=None,
        method="dopri5",
        hutchinson=False,
        options=None,
    ):
        """
        Diffusion model class without conditionals.
        """

        super(PopulationModelDiffusion, self).__init__()

        self.model = model
        self.sde = sde
        self.score_model = ScoreModel(
            model=self.model, sde=self.sde, hutchinson=hutchinson        
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

        return self.score_model.sample_sde(shape, steps=100) * self.scale + self.shift

    def log_prob(self, x, atol=1e-5, rtol=1e-5):
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
    """

    def __init__(
        self,
        model=None,
        sde=None,
        shift=None,
        scale=None,
        conditional_shift=None,
        conditional_scale=None,
        method="dopri5",
        options=None,
    ):

        super(PopulationModelDiffusionConditional, self).__init__()

        self.model = model
        self.sde = sde
        self.score_model = ScoreModel(model=self.model, sde=self.sde)
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
