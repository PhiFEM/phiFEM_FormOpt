from formopt import Model

from ufl import (
    jump,
    avg,
    div,
    CellDiameter,
    FacetArea,
    FacetNormal,
    TrialFunction,
    TestFunction,
    FacetNormal,
    Identity,
    Measure,
    conditional,
    as_vector,
    inner,
    outer,
    grad,
    dot,
    lt,
)


class LaplacianEnergy(Model):

    def __init__(self, dim, domain, space, path, rank_dim, mesh_data, vol):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        self.rank_dim = rank_dim

        cells_tags, facets_tags, self.ds_out = mesh_data
        self.dx = Measure("dx", domain=domain, subdomain_data=cells_tags)
        self.dS = Measure("dS", domain=domain, subdomain_data=facets_tags)

        self.vol = vol
        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        self.factor = 0.025
        self.stab_par = 1.0

    def pde(self, phi):

        h_T = CellDiameter(self.domain)
        nv = FacetNormal(self.domain)
        w = TrialFunction(self.space)
        phiw = phi * w
        v = TestFunction(self.space)
        phiv = phi * v

        a = inner(grad(phiw), grad(phiv)) * self.dx((1, 2))
        a -= inner(grad(phiw), nv) * phiv * self.ds_out
        Ga = (h_T**2) * div(grad(phiw)) * div(grad(phiv)) * self.dx(2)
        Ga += avg(h_T) * jump(grad(phiw), nv) * jump(grad(phiv), nv) * self.dS((2, 3))

        l = phiv * self.dx((1, 2))
        Gl = -(h_T**2) * div(grad(phiv)) * self.dx(2)

        return [(a - l + self.stab_par * (Ga - Gl), [])]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        return -self.factor * inner(grad(u), grad(u)) * self.dx((1, 2))

    def constraint(self, phi, U):
        # Volume constraint
        return [(1.0 / self.vol) * self.chi(phi) * self.dx]

    def derivative(self, phi, U, P):
        # Shape derivative components
        u = U[0]
        S0_J = self.zero_vec
        S1_J = (inner(grad(u), grad(u)) - 2.0 * u) * self.Id
        S1_J -= 2.0 * outer(grad(u), grad(u))
        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        return (S0_J, [S0_C]), (self.factor * S1_J, [S1_C])

    def bilinear_form(self, th, xi):
        B = dot(th, xi) * self.dx + inner(grad(th), grad(xi)) * self.dx
        return B, True
