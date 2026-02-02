from formopt import Model
from formopt import func_to_eval, qtty_to_eval

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
    le,
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

    def W(self, F):
        # Strain energy density
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

    def stress(self, F):
        return self.hatQ1(F) * F.T / det(F)


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

    def W(self, F):
        # Strain energy density
        eu = sym(F) - self.Id
        return 0.5 * inner(self.S(eu), eu)

    def hatQ1(self, F):
        # First Piola-Kirchhoff stress
        return self.S(sym(F) - self.Id)

    def hatQ2(self, F, dF):
        # First Piola-Kirchhoff tangent
        return self.S(sym(dF))

    def stress(self, F):
        return self.S(sym(F) - self.Id)


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

    def W(self, F):
        # Strain energy density
        return 0.5 * inner(self.S(self.E(F)), self.E(F))

    def hatQ1(self, F):
        # First Piola-Kirchhoff stress
        return F * self.S(self.E(F))

    def hatQ2(self, F, dF):
        # First Piola-Kirchhoff tangent
        return dF * self.S(self.E(F)) + F * self.S(sym(F.T * dF))

    def stress(self, F):
        return self.hatQ1(F) * F.T / det(F)


class Compliance(Model):

    def __init__(
        self,
        dim,
        domain,
        space,
        path,
        em,
        g_arr,
        dsg_arr,
        dir_bcs,
        alpha,
        pty=True,
        eps=1e-2,
    ):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        self.em = em

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)

        self.bc = dir_bcs
        self.u0 = lambda x: 0.0 * x[:dim]
        self.alpha = alpha
        self.pty = pty
        self.nN = -1
        self.sub = []

        self.g_arr = [[as_vector(g) for g in g_row] for g_row in g_arr]
        self.dsg_arr = dsg_arr

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.F = lambda w: self.Id + grad(w)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, eps)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        self.sign = lambda w: conditional(lt(w, 0.0), 1.0, -1.0)

    def lfunc(self, u, g_list, dsg_list):
        return sum([dot(g, u) * dsg for g, dsg in zip(g_list, dsg_list)])

    def dlfunc(self, u, du, g_list, dsg_list):
        return sum([dot(g, du) * dsg for g, dsg in zip(g_list, dsg_list)])

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    @func_to_eval("Displacement")
    def normDisplacement(self, phi, U, P):
        u = U[0]
        return self.sign(phi) * sqrt(dot(u, u))

    @func_to_eval("VonMises")
    def stressVonMises(self, phi, U, P):
        # Von Mises stress calculated using Cauchy stress
        u = U[0]
        sigma = self.em.stress(self.F(u))
        s = sigma - (1.0 / 3.0) * tr(sigma) * self.Id

        return self.chi(phi) * sqrt(1.5 * inner(s, sigma))

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

            pdes = []
            for g_row, dsg_row in zip(self.g_arr, self.dsg_arr):
                # Weak formulation of state problem
                Wf = self.A(phi) * self._DW(self.F(u), grad(v)) * self.dx
                for g, dsg in zip(g_row, dsg_row):
                    Wf -= dot(g, v) * dsg
                pdes.append((Wf, self.bc))

            return pdes
        else:
            u = Coefficient(self.space)
            v = TestFunction(self.space)
            du = TrialFunction(self.space)
            l = Constant(self.domain)
            F = self.F(u)

            pdes = []
            for g_row, dsg_row in zip(self.g_arr, self.dsg_arr):
                # Weak formulation of state problem
                Wf = self.A(phi) * self._DW(F, grad(v)) * self.dx
                for g, dsg in zip(g_row, dsg_row):
                    Wf -= l * dot(g, v) * dsg
                # Jacobian
                Jc = self.A(phi) * self._D2W(F, grad(du), grad(v)) * self.dx
                pdes.append((Wf, self.bc, Jc, u, self.u0, (l, self.nN)))

            return pdes

    def adjoint(self, phi, U):

        p = TrialFunction(self.space)
        q = TestFunction(self.space)

        adjs = []
        for g_row, dsg_row, u in zip(self.g_arr, self.dsg_arr, U):
            # Weak formulation of adjoint problem
            Wf = self.A(phi) * self._D2W(self.F(u), grad(p), grad(q)) * self.dx
            Wf += self.dlfunc(u, q, g_row, dsg_row)
            adjs.append((Wf, self.bc))

        return adjs

    def cost(self, phi, U):
        # Cost functional

        cf_list = []

        for g_row, dsg_row, u in zip(self.g_arr, self.dsg_arr, U):
            cf_list.append(self.lfunc(u, g_row, dsg_row))

        if self.pty:
            cf_list.append(self.alpha * self.chi(phi) * self.dx)

        return sum(cf_list)

    def constraint(self, phi, U):
        if self.pty:
            return []
        else:
            return [(1.0 / self.alpha) * self.chi(phi) * self.dx]

    def derivative(self, phi, U, P):

        S0_J = self.zero_vec

        S1_J_list = []
        for u, p in zip(U, P):
            F = self.F(u)

            Q1 = -grad(p).T * self.em.hatQ1(F)
            Q2 = -grad(u).T * self.em.hatQ2(F, grad(p))

            s1 = self._DW(F, grad(p)) * self.Id
            s1 += Q1 + Q2
            S1_J_list.append(self.A(phi) * s1)

        if self.pty:
            S1_J_list.append(self.alpha * self.chi(phi) * self.Id)
            return (S0_J, []), (sum(S1_J_list), [])
        else:
            S0_C = self.zero_vec
            S1_C = (1.0 / self.alpha) * self.chi(phi) * self.Id
            return (S0_J, [S0_C]), (sum(S1_J_list), [S1_C])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds

        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class Mechanism(Model):

    def __init__(
        self,
        dim,
        domain,
        space,
        path,
        em,
        g_in,
        ds_in,
        g_out,
        ds_out,
        dir_bcs,
        kappa,
        alpha,
        beta,
        eps=3e-3,
    ):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        self.em = em

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)

        self.bc = dir_bcs
        self.u0 = lambda x: 0.0 * x[:dim]
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.nN = -1
        self.sub = []

        self.g_in = [as_vector(_) for _ in g_in]
        self.ds_in = ds_in
        self.g_out = [as_vector(_) for _ in g_out]
        self.ds_out = ds_out

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.F = lambda w: self.Id + grad(w)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, eps)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        self.sign = lambda w: conditional(lt(w, 0.0), 1.0, -1.0)

    def jfunc(self, phi, grad_u):
        # j(grad u)
        # j does not depend on u
        volume = self.alpha * self.chi(phi)
        energy = self.beta * self.A(phi) * self.em.W(self.Id + grad_u)
        return volume + energy

    def djfunc(self, phi, grad_u):
        # Derivative w.r.t. grad_u
        return self.beta * self.A(phi) * self.em.hatQ1(self.Id + grad_u)

    def lfunc(self, u):

        l_in = sum([dot(g, u) * dsg for g, dsg in zip(self.g_in, self.ds_in)])
        l_out = sum([dot(g, u) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

        return l_in + self.kappa * l_out

    def dlfunc(self, du):

        l_in = sum([dot(g, du) * dsg for g, dsg in zip(self.g_in, self.ds_in)])
        l_out = sum([dot(g, du) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

        return l_in + self.kappa * l_out

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    @qtty_to_eval("InDisp")
    def InDisp(self, phi, U, P):
        u = U[0]
        return sum([dot(g, u) * dsg for g, dsg in zip(self.g_in, self.ds_in)])

    @qtty_to_eval("OutDisp")
    def OutDisp(self, phi, U, P):
        u = U[0]
        return sum([dot(g, u) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

    @func_to_eval("Displacement")
    def normDisplacement(self, phi, U, P):
        u = U[0]
        return self.sign(phi) * sqrt(dot(u, u))

    @func_to_eval("VonMises")
    def stressVonMises(self, phi, U, P):
        # Von Mises stress calculated using Cauchy stress
        u = U[0]
        sigma = self.em.stress(self.F(u))
        s = sigma - (1.0 / 3.0) * tr(sigma) * self.Id

        return self.chi(phi) * sqrt(1.5 * inner(s, sigma))

    def _DW(self, phi, F, dF):
        # First derivative of strain energy density
        return self.A(phi) * inner(dF, self.em.hatQ1(F))

    def _D2W(self, phi, F, dF1, dF2):
        # Second derivative of strain energy density
        return self.A(phi) * inner(dF1, self.em.hatQ2(F, dF2))

    def pde(self, phi):

        if self.em.lineal:
            u = TrialFunction(self.space)
            v = TestFunction(self.space)

            # Weak formulation of state problem
            Wf = self._DW(phi, self.F(u), grad(v)) * self.dx

            for g, dsg in zip(self.g_in, self.ds_in):
                Wf -= dot(g, v) * dsg

            for g, dsg in zip(self.g_out, self.ds_out):
                Wf -= dot(g, v) * dsg

            return [(Wf, self.bc)]

        else:
            u = Coefficient(self.space)
            v = TestFunction(self.space)
            du = TrialFunction(self.space)
            l = Constant(self.domain)
            F = self.F(u)

            # Weak formulation of state problem
            Wf = self._DW(phi, F, grad(v)) * self.dx

            for g, dsg in zip(self.g_in, self.ds_in):
                Wf -= l * dot(g, v) * dsg

            for g, dsg in zip(self.g_out, self.ds_out):
                Wf -= l * dot(g, v) * dsg

            # Jacobian
            Jc = self._D2W(phi, F, grad(du), grad(v)) * self.dx

            return [(Wf, self.bc, Jc, u, self.u0, (l, self.nN))]

    def adjoint(self, phi, U):

        p = TrialFunction(self.space)
        q = TestFunction(self.space)
        u = U[0]

        # Weak formulation of adjoint problem
        Wf = self._D2W(phi, self.F(u), grad(p), grad(q)) * self.dx
        Wf += inner(self.djfunc(phi, grad(u)), grad(q)) * self.dx
        Wf += self.dlfunc(q)

        return [(Wf, self.bc)]

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        return self.jfunc(phi, grad(u)) * self.dx + self.lfunc(u)

    def constraint(self, phi, U):
        return []

    def derivative(self, phi, U, P):
        u = U[0]
        p = P[0]
        F = self.F(u)

        S0_J = self.zero_vec

        Q1 = -grad(p).T * self.A(phi) * self.em.hatQ1(F)
        Q2 = -grad(u).T * self.A(phi) * self.em.hatQ2(F, grad(p))
        Q3 = -grad(u).T * self.djfunc(phi, grad(u))

        S1_J = (self.jfunc(phi, grad(u)) + self._DW(phi, F, grad(p))) * self.Id
        S1_J += Q1 + Q2 + Q3

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds

        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False
