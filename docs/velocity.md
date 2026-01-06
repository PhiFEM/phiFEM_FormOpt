# Velocity class

The `Velocity` class sets up and solves the following weak formulation: Find $\theta \in \mathbb{H}$ such that 
$$
B(\theta, \xi) = - \int_{\mathcal{D}} S_0\cdot\xi + S_1 : D\xi \quad\forall \; \xi \in \mathbb{H},
$$
where

- $\mathbb{H} = H^1(\mathcal{D})^d$ or $\mathbb{H} = H_0^1(\mathcal{D})^d$;
- $B$ is a bilinear form defined in the model class;
- $S_0$ and $S_1$ are shape derivative components of either
  the cost functional $J$ or the Lagrangian $L$ (if there are constraints).

The velocity field $\theta$ is used to update the level set function.

---

## Details

::: code.formopt.Velocity
