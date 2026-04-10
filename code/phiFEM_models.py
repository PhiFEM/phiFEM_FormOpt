from formopt import Model, qtty_to_eval

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
    TrialFunctions,
    TestFunctions,
    sym,
    nabla_div,
    dot,
)


class ComplianceElasticity(Model):
    def __init__(self, dim, domain, mixed_space, path, rank_dim, g, dsg, bcs, volume):

        self.mixed_space = mixed_space

        self.dim = dim
        self.domain = domain
        self.space, self.map = mixed_space.sub(0).collapse()
        self.path = path
        self.rank_dim = rank_dim

        self.dx = Measure("dx", domain=self.domain)
        self.dS = Measure("dS", domain=self.domain)
        self.ds_out = Measure("ds", domain=self.domain)
        self.ds = Measure("ds", domain=self.domain)

        E, nu = 1.0, 0.3
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)

        self.g = as_vector(g)
        self.dsg = dsg
        self.bcs = bcs
        self.sub = []

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda v: sym(grad(v))
        self.sigma = lambda v: (
            lmbda * nabla_div(v) * self.Id + 2.0 * mu * self.epsilon(v)
        )
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        self.vol = volume
        self.factor = 25.0

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    def pde(self, phi):
        (u, y, p) = TrialFunctions(self.mixed_space)
        (v, z, q) = TestFunctions(self.mixed_space)

        hT = CellDiameter(self.domain)
        nf = FacetNormal(self.domain)

        yp = dot(y, grad(phi)) + (p * phi / hT)
        zq = dot(z, grad(phi)) + (q * phi / hT)

        gamma_u = 1.0
        gamma_p = 1.0
        sigma_D = 20.0
        gamma_div = 1.0

        su, sv = self.sigma(u), self.sigma(v)
        ev = self.epsilon(v)

        # Weak formulation
        wf = inner(su, ev) * self.dx((1, 2))
        wf += dot(dot(y, nf), v) * self.ds_out
        wf += gamma_u * inner(y + su, z + sv) * self.dx(2)
        wf += gamma_p * (hT**-2) * dot(yp, zq) * self.dx(2)

        # Estabilization
        wf += sigma_D * avg(hT) * inner(jump(su, nf), jump(sv, nf)) * self.dS((2, 3))
        wf += gamma_div * inner(div(y), div(z)) * self.dx(2)

        # Force application
        wf -= dot(self.g, v) * self.dsg

        return [(wf, self.bcs)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        su, eu = self.sigma(u), self.epsilon(u)
        return self.factor * self.chi(phi) * inner(su, eu) * self.dx

    def constraint(self, phi, U):
        # Volume constraint
        return [(1.0 / self.vol) * self.chi(phi) * self.dx]

    def derivative(self, phi, U, P):
        # Shape derivative components
        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        S0_J = self.zero_vec
        S1_J = 2.0 * grad(u).T * su
        S1_J -= inner(su, eu) * self.Id
        S1_J *= self.factor * self.chi(phi)

        # volume constraint
        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        return (S0_J, [S0_C]), (S1_J, [S1_C])

    def bilinear_form(self, th, xi):
        # Weighted H1-norm with penalty terms
        nv = FacetNormal(self.domain)

        B = 0.1 * dot(th, xi) * self.dx
        B += inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class LaplacianEnergy(Model):

    def __init__(self, dim, domain, space, path, rank_dim, vol):

        self.mixed_space = False

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        self.rank_dim = rank_dim

        self.dx = Measure("dx", domain=domain)
        self.dS = Measure("dS", domain=domain)
        self.ds_out = Measure("ds", domain=domain)

        self.vol = vol
        self.zero_vec = as_vector(dim * [0.0])
        self.One = (1.0 / dim) * dot(as_vector(dim * [1.0]), as_vector(dim * [1.0]))
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
        return -self.factor * self.chi(phi) * inner(grad(u), grad(u)) * self.dx

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

        return (S0_J, [S0_C]), (self.factor * self.chi(phi) * S1_J, [S1_C])

    def bilinear_form(self, th, xi):
        # H1-norm
        B = dot(th, xi) * self.dx + 0.1 * inner(grad(th), grad(xi)) * self.dx
        return B, True
