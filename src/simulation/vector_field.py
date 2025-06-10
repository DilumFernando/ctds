from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, jvp, vmap

from src.systems.density_path import DensityPath, LinearGaussianInterpolation
from src.systems.distributions import Gaussian
from src.utils.misc import cuda_profile
from src.utils.nn import FeedForward, GaussianFourierEncoder

################
# Vector Field #
################

class VectorField(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        pass

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        return self.drift(x, t)

    def autograd_divergence(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            div: (batch_size, 1)
        """
        # Create a pseudo-batch dimension
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        t = t.unsqueeze(1)  # (batch_size, 1, 1)
        
        # Compute the Jacobian of the vector field with respect to x
        batched_jacs = vmap(jacrev(self, argnums=0))(
            x, t
        )  # (batch_size, 1, dim, 1, dim)
        batched_jacs = batched_jacs.squeeze(1, 3)  # (batch_size, dim, dim)
        # (batch_size, 1)
        return torch.einsum("bii->b", batched_jacs).unsqueeze(-1)

    def hutch_numeric_divergence(
        self, x: Tensor, t: Tensor, delta: float = 0.01
    ) -> Tensor:
        """
        Unbiased estimate of divergence (when delta -> 0). Growing linearly with dim.
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
            delta: used for numerical approximation of the Jacobian-vector product.
        Returns:
            div: (batch_size, 1)
        """
        noise = torch.randn_like(x)  # (bs, dim)
        x_fwd = x + delta * noise
        x_bwd = x - delta * noise
        v_fwd = self(x_fwd, t)  # (bs, dim)
        v_bwd = self(x_bwd, t)  # (bs, dim)
        diff = (v_fwd - v_bwd) / (2 * delta)  # (bs, dim)
        return torch.einsum("bi,bi->b", noise, diff).unsqueeze(-1)

    def hutch_jvp_divergence(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Hutchinson trace estimator; grows linearly with dim.
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            div: (batch_size, 1)
        """
        noise = torch.rand_like(x)  # (bs, dim)

        def x_only(x: Tensor) -> Tensor:
            return self(x, t)

        _, cjvp = jvp(x_only, (x,), (noise,))  # (bs, dim)
        return torch.einsum("bi,bi->b", noise, cjvp).unsqueeze(-1)  # (bs, 1)

    @cuda_profile
    def divergence(self, x: Tensor, t: Tensor, mode: str) -> Tensor:
        """
        Returns divergence with respect to x
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
            mode: "hutch_numeric" | "hutch_jvp" | "autograd"
        Returns:
            div: (batch_size, 1)
        """
        if mode == "hutch_numeric":
            return self.hutch_numeric_divergence(x, t)
        elif mode == "hutch_jvp":
            return self.hutch_jvp_divergence(x, t)
        elif mode == "autograd":
            return self.autograd_divergence(x, t)


class ZeroVectorField(VectorField):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        xt: (batch_size, dim)
        t: (batch_size, 1)
        """
        return torch.zeros(x.shape[0], self.out_dim).to(x)


class SumVectorField(VectorField):
    def __init__(self, drifts: List[VectorField]):
        super().__init__()
        self.drifts = drifts

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        xt: (batch_size, dim)
        t: (batch_size, 1)
        """
        output = torch.zeros_like(x)
        for drift in self.drifts:
            output += drift(x, t)
        return output


class LinearGaussianInterpolationVectorField(VectorField):
    def __init__(self, source: Gaussian, target: Gaussian):
        super().__init__()
        self.density_path = LinearGaussianInterpolation(source, target)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        xt: (batch_size, dim)
        t: (batch_size, 1)
        """
        mu_t = torch.vmap(self.density_path.get_mu_t)(t)  # (batch_size, dim)
        sigma_t_cholesky = torch.vmap(self.density_path.get_sigma_t_cholesky)(
            t
        )  # (batch_size, dim, dim)
        A = torch.bmm(
            torch.linalg.inv(sigma_t_cholesky), (x - mu_t).unsqueeze(-1)
        )  # (batch_size, dim, 1)
        dt_mu_t = torch.vmap(self.density_path.get_dt_mu_t)(t)  # (batch_size, dim)
        dt_sigma_t_cholesky = vmap(jacrev(self.density_path.get_sigma_t_cholesky))(
            t
        ).squeeze(
            -1
        )  # (batch_size, dim, dim)
        return torch.bmm(dt_sigma_t_cholesky, A).squeeze(-1) + dt_mu_t
    
class MLPVectorField(VectorField):
    """
    MLP-parameterized vector field R^{data_dim + time_embed_dim} -> R^{data_dim}
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dims: List[int],
        use_fourier: bool,
        x_fourier_dim: Optional[int] = None,
        x_fourier_sigma: Optional[float] = None,
        t_fourier_dim: Optional[int] = None,
        t_fourier_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert x_fourier_dim is not None and t_fourier_dim is not None
            assert x_fourier_sigma is not None and t_fourier_sigma is not None
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + hidden_dims + [data_dim])
            self.x_encoder = GaussianFourierEncoder(data_dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def fwd(x: Tensor, t: Tensor) -> Tensor:
                x_encoding = self.x_encoder(x)
                t_encoding = self.t_encoder(t)
                xt_encoding = torch.cat([x_encoding, t_encoding], dim=-1)
                return self.mlp(xt_encoding)

            self.net = fwd
        else:
            self.mlp = FeedForward([data_dim + 1] + hidden_dims + [data_dim])

            def fwd(x: Tensor, t: Tensor) -> Tensor:
                x_t = torch.cat([x, t], dim=-1)
                return self.mlp(x_t)

            self.net = fwd

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(x, t)


####################
# Interpolant Path #
####################

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.zeros(1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1)), torch.ones(1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
    
class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.ones(1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1)), torch.zeros(1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
    

class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return torch.ones_like(t)

class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return - torch.ones_like(t)
    
#############################
# Conditional Vector Fields #
#############################
    
class ConditionalVectorField(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def drift(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            c: (batch_size, c_dim)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        pass

    def forward(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            c: (batch_size, c_dim)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        return self.drift(x, c, t)

    def autograd_divergence(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            c: (batch_size, c_dim)
            t: (batch_size, 1)
        Returns:
            div: (batch_size, 1)
        """
        # Create a pseudo-batch dimension
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        t = t.unsqueeze(1)  # (batch_size, 1, 1)
        c = c.unsqueeze(1)  # (batch_size, 1, dim)
        
        # Compute the Jacobian of the vector field with respect to x
        batched_jacs = vmap(jacrev(self, argnums=0))(
            x, c, t
        )  # (batch_size, 1, dim, 1, dim)
        batched_jacs = batched_jacs.squeeze(1, 3)  # (batch_size, dim, dim)
        # (batch_size, 1)
        return torch.einsum("bii->b", batched_jacs).unsqueeze(-1)

    def hutch_numeric_divergence(
        self, x: Tensor, c: Tensor, t: Tensor, delta: float = 0.01
    ) -> Tensor:
        """
        Unbiased estimate of divergence (when delta -> 0). Growing linearly with dim.
        Args:
            x: (batch_size, dim)
            c: (batch_size, c_dim)
            t: (batch_size, 1)
            delta: used for numerical approximation of the Jacobian-vector product.
        Returns:
            div: (batch_size, 1)
        """
        noise = torch.randn_like(x)  # (bs, dim)
        x_fwd = x + delta * noise
        x_bwd = x - delta * noise
        v_fwd = self(x_fwd, c, t)  # (bs, dim)
        v_bwd = self(x_bwd, c, t)  # (bs, dim)
        diff = (v_fwd - v_bwd) / (2 * delta)  # (bs, dim)
        return torch.einsum("bi,bi->b", noise, diff).unsqueeze(-1)

    def hutch_jvp_divergence(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        """
        Hutchinson trace estimator; grows linearly with dim.
        Args:
            x: (batch_size, dim)
            c: (batch_size, c_dim)
            t: (batch_size, 1)
        Returns:
            div: (batch_size, 1)
        """
        noise = torch.rand_like(x)  # (bs, dim)

        def x_only(x: Tensor) -> Tensor:
            return self(x, c, t)

        _, cjvp = jvp(x_only, (x,), (noise,))  # (bs, dim)
        return torch.einsum("bi,bi->b", noise, cjvp).unsqueeze(-1)  # (bs, 1)

    @cuda_profile
    def divergence(self, x: Tensor, c: Tensor, t: Tensor, mode: str) -> Tensor:
        """
        Returns divergence with respect to x
        Args:
            x: (batch_size, dim)
            c: (batch_size, c_dim)
            t: (batch_size, 1)
            mode: "hutch_numeric" | "hutch_jvp" | "autograd"
        Returns:
            div: (batch_size, 1)
        """
        if mode == "hutch_numeric":
            return self.hutch_numeric_divergence(x, c, t)
        elif mode == "hutch_jvp":
            return self.hutch_jvp_divergence(x, c, t)
        elif mode == "autograd":
            return self.autograd_divergence(x, c, t)

class ZeroConditionalVectorField(ConditionalVectorField):
    def drift(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        """
        xt: (batch_size, dim)
        t: (batch_size, 1)
        """
        return torch.zeros_like(x)  # (batch_size, dim) 

class ConditionalMLPVectorField(ConditionalVectorField):
    """
    MLP-parameterized vector field R^{data_dim + time_embed_dim} -> R^{data_dim}
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dims: List[int],
        conditioning_dim: int = 1,
        use_fourier: bool = False,
        x_fourier_dim: Optional[int] = None,
        x_fourier_sigma: Optional[float] = None,
        c_fourier_dim: Optional[int] = None,
        c_fourier_sigma: Optional[float] = None,
        t_fourier_dim: Optional[int] = None,
        t_fourier_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert x_fourier_dim is not None and c_fourier_dim is not None and t_fourier_dim is not None
            assert x_fourier_sigma is not None and c_fourier_sigma is not None and t_fourier_sigma is not None
            input_dim = x_fourier_dim + c_fourier_dim + t_fourier_dim
            self.x_encoder = GaussianFourierEncoder(data_dim, x_fourier_dim, x_fourier_sigma)
            self.c_encoder = GaussianFourierEncoder(conditioning_dim, c_fourier_dim, c_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)
            self.mlp = FeedForward([input_dim] + hidden_dims + [data_dim])
        else:
            self.mlp = FeedForward([data_dim + conditioning_dim + 1] + hidden_dims + [data_dim])

    def drift(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        if self.use_fourier:
            x_encoding = self.x_encoder(x)
            c_encoding = self.c_encoder(c)
            t_encoding = self.t_encoder(t)
            xct_encoding = torch.cat([x_encoding, c_encoding, t_encoding], dim=-1)
            return self.mlp(xct_encoding)
        else:
            x_c_t = torch.cat([x, c, t], dim=-1)
            return self.mlp(x_c_t)