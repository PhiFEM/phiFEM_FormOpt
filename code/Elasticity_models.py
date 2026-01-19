from formopt import Model

from ufl import (
    TrialFunction,
    TestFunction,
    FacetNormal,
    Identity,
    Measure,
    SpatialCoordinate,
    Coefficient,
    conditional,
    indices,
    as_vector,
    inner,
    outer,
    grad,
    sym,
    dot,
    lt,
    pi,
    cos,
    sqrt,
    nabla_div,
    Constant,
    det,
    inv,
    tr,
    ln,
)


class MRcomponents:
    """
    Mooney-Rivlin material model components
    """

    def __init__(self, mu=76.923, lmbda=115.385, dim=2):

        self.dim = dim

        delta = min([0.5 * mu, 0.25 * lmbda]) / 100.0
        print("Mooney-Rivlin delta =", delta)
        alpha = 0.5 * mu - delta
        beta = 0.25 * lmbda - delta
        gamma = mu + 0.5 * lmbda

        self.a1 = 2.0 * alpha
        self.a2 = 2.0 * delta
        self.Z = lambda x: beta * x**2 - gamma * ln(x)
        self.DZ = lambda x: 2.0 * beta * x - gamma / x
        self.D2Z = lambda x: 2.0 * beta + gamma / (x**2)

        self.Id = Identity(dim)
        self.lineal = False

    def Cof(self, F):
        if self.dim == 2:
            return tr(F) * self.Id - F.T
        if self.dim == 3:
            cf = 0.5 * (tr(F) ** 2 - tr(F * F)) * self.Id
            cf -= tr(F) * F
            cf += F * F
            return cf.T

    def DCof(self, F, dF):
        if self.dim == 2:
            return tr(dF) * self.Id - dF.T
        if self.dim == 3:
            dc = (tr(dF) * tr(F) - tr(F * dF)) * self.Id
            dc -= tr(dF) * F.T + tr(F) * dF.T
            dc += F.T * dF.T + dF.T * F.T
            return dc

    def W(self, u):
        # Strain energy density
        F = self.Id + grad(u)
        J, H = det(F), self.Cof(F)
        wf = 0.5 * self.a1 * inner(F, F)
        wf += 0.5 * self.a2 * inner(H, H)
        wf += self.Z(J)

        return wf

    def M(self, F, B):
        # M(F; B)
        if self.dim == 2:
            return tr(B) * self.Id - B.T
        if self.dim == 3:
            mb = tr(F) * tr(B) * self.Id
            mb -= inner(F.T, B) * self.Id
            mb -= tr(F) * B.T + tr(B) * F.T
            mb += F.T * B.T + B.T * F.T
            return mb

    def DM(self, B, dF):
        # DM(F; B)(dF) is independent of F in 2D and 3D
        if self.dim == 2:
            return 0.0 * self.Id
        if self.dim == 3:
            return self.M(dF, B)

    def hatQ1(self, F):
        J, H = det(F), self.Cof(F)

        hq1 = self.a1 * F
        hq1 += self.a2 * self.M(F, H)
        hq1 += self.DZ(J) * H

        return hq1

    def hatQ2(self, F, dF):
        J, H = det(F), self.Cof(F)

        hq2 = self.DZ(J) * self.M(F, dF)
        hq2 += self.a2 * self.DM(H, dF)
        hq2 += self.D2Z(J) * inner(H, dF) * H
        hq2 += self.a1 * dF
        hq2 += self.a2 * self.M(F, self.DCof(F, dF))

        return hq2


class Hookecomponents:
    """
    Linear Hooke material model components
    """

    def __init__(self, Ym=200.0, Pr=0.3, dim=2):
        # Ym : Young's modulus
        # Pr : Poisson's ratio
        lmbda = Ym * Pr / (1.0 + Pr) / (1.0 - 2.0 * Pr)
        mu = Ym / 2.0 / (1.0 + Pr)
        self.Id = Identity(dim)
        self.S = lambda M: lmbda * tr(M) * self.Id + 2.0 * mu * M
        self.lineal = True

    def W(self, u):
        # Strain energy density
        eu = sym(grad(u))
        return 0.5 * inner(self.S(eu), eu)

    def hatQ1(self, F):
        # First Piola-Kirchhoff stress
        return self.S(sym(F) - self.Id)

    def hatQ2(self, F, dF):
        # First Piola-Kirchhoff tangent
        return self.S(sym(dF))


class SVKcomponents:
    """
    Saint Venant-Kirchhoff material model components

    The derivatives of strain energy density W(F)
    can be expressed as:
        - W'(F)(dF) = dF : P
        - W''(F)(dF1, dF2) = dF1 : DP(F)(dF2)
    where P is the first Piola-Kirchhoff stress
    and DP(F)(dF) is the first derivative of P at F
    in the direction dF.
    """

    def __init__(self, Ym=200.0, Pr=0.3, dim=2):
        # Ym : Young's modulus
        # Pr : Poisson's ratio
        lmbda = Ym * Pr / (1.0 + Pr) / (1.0 - 2.0 * Pr)
        mu = Ym / 2.0 / (1.0 + Pr)
        Id = Identity(dim)
        self.S = lambda M: lmbda * inner(M, Id) * Id + 2.0 * mu * M
        self.E = lambda M: 0.5 * (M.T * M - Id)
        self.lineal = False

    def W(self, u):
        # Strain energy density
        F = self.Id + grad(u)
        return 0.5 * inner(self.S(self.E(F)), self.E(F))

    def hatQ1(self, F):
        # First Piola-Kirchhoff stress
        return F * self.S(self.E(F))

    def hatQ2(self, F, dF):
        # First Piola-Kirchhoff tangent
        return dF * self.S(self.E(F)) + F * self.S(sym(F.T * dF))


class Compliance:

    def __init__(self, g, ds_g):
        self.g = [as_vector(gi) for gi in g]
        self.ds_g = ds_g

    def jfunc(self, u):
        return 0.0

    def djfunc(self, u, du):
        return 0.0

    def lfunc(self, u):
        return sum([dot(gi, u) * dgi for gi, dgi in zip(self.g, self.ds_g)])

    def dlfunc(self, u, du):
        return sum([dot(gi, du) * dgi for gi, dgi in zip(self.g, self.ds_g)])


class Gripping:

    def __init__(self, k, ds_g):
        self.k = [as_vector(ki) for ki in k]
        self.ds_g = ds_g

    def jfunc(self, u):
        return 0.0

    def djfunc(self, u, du):
        return 0.0

    def lfunc(self, u):
        return sum([dot(ki, u) * dgi for ki, dgi in zip(self.k, self.ds_g)])

    def dlfunc(self, u, du):
        return sum([dot(ki, du) * dgi for ki, dgi in zip(self.k, self.ds_g)])


class Elasticity(Model):

    def __init__(
        self, dim, domain, space, path, em, cf, g, ds_g, dir_bcs, alpha, eps=1e-3
    ):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        self.em = em
        self.cf = cf
        self.fcs_to_eval = [self.stressVonMises]
        self.fte_names = ["VonMises"]

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)

        self.bc = dir_bcs
        self.u0 = lambda x: 0.0 * x[:dim]
        self.alpha = alpha
        self.nN = 4

        self.g = [as_vector(gi) for gi in g]
        self.ds_g = ds_g

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.F = lambda w: self.Id + grad(w)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, eps)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)

    def stressVonMises(self, phi, U, P):

        u = U[0]
        F = self.F(u)
        sigma = self.em.hatQ1(F) * F.T / det(F)
        s = sigma - (1.0 / 3.0) * tr(sigma) * self.Id

        return sqrt(1.5 * inner(s, s))

    def _DW(self, F, dF):
        # First derivative of strain energy density
        return inner(dF, self.em.hatQ1(F))

    def _D2W(self, F, dF1, dF2):
        # Second derivative of strain energy density
        return inner(dF1, self.em.hatQ2(F, dF2))

    def pde(self, phi):

        if self.em.lineal:
            u = TrialFunction(self.space)
            v = TestFunction(self.space)

            # Weak formulation of state problem
            Wf = self.A(phi) * self._DW(self.F(u), grad(v)) * self.dx
            for gi, dgi in zip(self.g, self.ds_g):
                Wf -= dot(gi, v) * dgi

            return [(Wf, self.bc)]
        else:
            u = Coefficient(self.space)
            v = TestFunction(self.space)
            du = TrialFunction(self.space)
            l = Constant(self.domain)
            F = self.F(u)

            # Weak formulation of state problem
            Wf = self.A(phi) * self._DW(F, grad(v)) * self.dx
            for gi, dgi in zip(self.g, self.ds_g):
                Wf -= l * dot(gi, v) * dgi
            # Jacobian
            Jc = self.A(phi) * self._D2W(F, grad(du), grad(v)) * self.dx

            return [(Wf, self.bc, Jc, u, self.u0, l, (0.1, self.nN))]

    def adjoint(self, phi, U):

        u = U[0]
        p = TrialFunction(self.space)
        q = TestFunction(self.space)

        F = self.F(u)

        # Weak formulation of adjoint problem
        Wf = self.A(phi) * self._D2W(F, grad(p), grad(q)) * self.dx
        Wf += self.cf.djfunc(u, q) * self.dx + self.cf.dlfunc(u, q)

        return [(Wf, self.bc)]

    def cost(self, phi, U):

        u = U[0]

        # Cost functional
        cf = self.cf.jfunc(u) * self.dx
        cf += self.cf.lfunc(u)
        cf += self.alpha * self.chi(phi) * self.dx

        return cf

    def constraint(self, phi, U):
        return []

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]

        F = self.F(u)

        Q1 = -grad(p).T * self.em.hatQ1(F)
        Q2 = -grad(u).T * self.em.hatQ2(F, grad(p))

        S1_J = (self.cf.jfunc(u) + self._DW(F, grad(p))) * self.Id
        S1_J += Q1 + Q2
        S1_J *= self.A(phi)
        S1_J += self.alpha * self.chi(phi) * self.Id

        S0_J = self.zero_vec

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds

        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False
