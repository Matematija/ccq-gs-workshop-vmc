from typing import NamedTuple, Callable

from math import sqrt

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Scalar, PyTree, Key


class RBMParams(NamedTuple):

    a: Float[Array, "n_visible"]  # Visible biases
    b: Float[Array, "n_hidden"]  # Hidden biases
    W: Float[Array, "n_visible n_hidden"]  # Weights

    @classmethod
    def initialize(cls, n_visible: int, n_hidden: int, key: Key, sigma: float = 0.01):

        key_a, key_b, key_W = jax.random.split(key, 3)
        a = sigma * jax.random.normal(key_a, shape=(n_visible,))
        b = sigma * jax.random.normal(key_b, shape=(n_hidden,))
        W = sigma * jax.random.normal(key_W, shape=(n_visible, n_hidden))

        return cls(a, b, W)


def log_cosh(x: Array) -> Array:
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def eval_rbm(params: RBMParams, s: Float[Array, "n_visible"]) -> Scalar:
    a, b, W = params
    return a @ s + log_cosh(b + W.T @ s).sum()


def eloc_zz_1d(x: Float[Array, "Lx"]):
    return x @ jnp.roll(x, -1, axis=1)


def eloc_zz_2d(x: Float[Array, "Lx Ly"]):
    return x @ jnp.roll(x, -1, axis=0) + x @ jnp.roll(x, -1, axis=1)


# (...) Generalization to arbitrary dimensions


def eloc_zz(x: Float[Array, "Lx Ly"]):
    big_dot = lambda a, b: jnp.tensordot(a, b, axes=a.ndim)  # full contraction
    return sum(big_dot(x, jnp.roll(x, -1, axis=i)) for i in range(x.ndim))


def eloc_x(logpsi: Callable, params: PyTree, x: Float[Array, "..."]):

    conn = jnp.broadcast_to(x.ravel(), (x.size, x.size))
    conn = conn.at[jnp.diag_indices_from(conn)].multiply(-1).reshape(-1, *x.shape)
    mels = jnp.ones(x.size)

    logpsi_conn = jax.vmap(logpsi, in_axes=(None, 0))(params, conn)
    logpsi_x = logpsi(params, x)

    return mels @ jnp.exp(logpsi_conn - logpsi_x)


def tfim_local_energy(
    logpsi: Callable, params: PyTree, x: Float[Array, "n_visible"], h: Scalar, J: Scalar = 1.0
) -> Scalar:
    return -J * eloc_zz(x) - h * eloc_x(logpsi, params, x)


################################################################


def energy_value_and_grad(
    eloc: Callable, logpsi: Callable, params: PyTree, samples: Float[Array, "n_samples ..."]
):
    """Compute the gradient of the energy expectation value.

    Args:
        eloc: Function to compute the local energy.
        logpsi: Function to compute the logarithm of the wavefunction.
        params: Parameters of the neural network wavefunction.
        samples: Samples drawn from the probability distribution.

    Returns:
        Gradient of the energy expectation value with respect to parameters.
    """

    params_flat, re = ravel_pytree(params)

    O = jax.vmap(jax.grad(logpsi), in_axes=(None, 0))(params, samples)
    local_energies = jax.vmap(lambda x: eloc(logpsi, params, x))(samples)
    energy = jnp.mean(local_energies)

    n_samples = samples.shape[0]
    O_ = O / sqrt(n_samples)
    e_ = (local_energies - energy) / sqrt(n_samples)
    energy_grad = 2 * jnp.real(O_ @ e_)

    return energy.real, energy_grad


def energy_value_and_natural_grad(
    eloc: Callable,
    logpsi: Callable,
    params: RBMParams,
    samples: Float[Array, "n_samples ..."],
    eps=1e-5,
):

    O = jax.vmap(jax.grad(logpsi), in_axes=(None, 0))(params, samples)

    local_energies = jax.vmap(lambda x: eloc(logpsi, params, x))(samples)
    energy = jnp.mean(local_energies)

    n_samples = samples.shape[0]
    O_ = O / sqrt(n_samples)
    e_ = (local_energies - energy) / sqrt(n_samples)
    energy_grad = 2 * jnp.real(O_ @ e_)

    S = jnp.conj(O_).T @ O_
    S = S.at[jnp.diag_indices_from(S)].add(eps)
    natural_grad = jnp.linalg.solve(S, energy_grad)

    return energy.real, natural_grad
