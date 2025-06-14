# Clase Reinit

La clase `Reinit` implementa un método iterativo para resolver la ecuación de re-inicialización:

$$
\frac{\partial \phi}{\partial \tau} + \text{sign}(\phi_0)(|\nabla \phi| - 1) = 0
$$

::: code.distributed.Reinit

---

## Comentarios adicionales

Este método se basa en un esquema explícito en tiempo y puede ser sensible al tamaño de paso `dt`. Asegúrate de que `dt` satisfaga la condición CFL para estabilidad.

También puede interesarte:
- [Documentación del módulo `distributed`](../code/distributed.md)
- [Notas teóricas sobre EDPs](../notas/edps.md)

