# Reinit class

The `Reinit` class solve the Eikonal equation

$$
\begin{aligned}
    \partial_t \varphi - h^2 \nabla^2 \varphi &= S(\phi)(1 - |\nabla \varphi|)
    && x\in \mathcal{D},\, t>0, \\
    \partial_n \varphi &= 0
    && x\in \partial\mathcal{D},\, t>0, \\
	\varphi(0, x) &= \phi(x)
	&& x\in \overline{\mathcal{D}},
\end{aligned}
$$

where $\phi$ is the current level set function, $h$ is the mesh diameter, and

$$
S(\phi) = \frac{\phi}{\sqrt{\phi^2 + h^2}}.
$$

---

::: code.distributed.Reinit



