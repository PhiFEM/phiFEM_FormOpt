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

    def Dstress(self, F, dF):
        Ds1 = (self.hatQ1(F) * dF.T + self.hatQ2(F, dF) * F.T) / det(F)
        Ds2 = (inner(self.Cof(F), dF) * self.hatQ1(F) * F.T) / (det(F) ** 2)
        return Ds1 - Ds2


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

    def hatQ3(self, L, F):
        return self.S(sym(L))

    def stress(self, F):
        return self.S(sym(F) - self.Id)

    def Dstress(self, F, dF):
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
        self.Id = Identity(dim)
        self.S = lambda M: lmbda * inner(M, self.Id) * self.Id + 2.0 * mu * M
        self.E = lambda M: 0.5 * (M.T * M - self.Id)
        self.lineal = False

    def W(self, F):
        # Strain energy density
        return 0.5 * inner(self.S(self.E(F)), self.E(F))

    def Cof(self, F):
        if self.dim == 2:
            return tr(F) * self.Id - F.T
        if self.dim == 3:
            cf = 0.5 * (tr(F) ** 2 - tr(F * F)) * self.Id
            cf -= tr(F) * F
            cf += F * F
            return cf.T

    def hatQ1(self, F):
        # First Piola-Kirchhoff stress
        return F * self.S(self.E(F))

    def hatQ2(self, F, dF):
        # First Piola-Kirchhoff tangent
        return dF * self.S(self.E(F)) + F * self.S(sym(F.T * dF))

    def hatQ3(self, L, F):
        q3 = L.T * self.hatQ1(F) / det(F)
        q3 -= inner(L, self.hatQ1(F) * F.T) * self.Cof(F) / (det(F) ** 2)
        q3 += L * F * self.S(self.E(F))
        q3 += F * self.S(sym(F.T * L * F))
        return q3

    def stress(self, F):
        return self.hatQ1(F) * F.T / det(F)

    def Dstress(self, F, dF):
        Ds1 = (self.hatQ1(F) * dF.T + self.hatQ2(F, dF) * F.T) / det(F)
        Ds2 = (inner(self.Cof(F), dF) * self.hatQ1(F) * F.T) / (det(F) ** 2)
        return Ds1 - Ds2


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

    def update_gs(self, g_arr):
        self.g_arr = [[as_vector(g) for g in g_row] for g_row in g_arr]

    def lfunc(self, u, g_list, dsg_list):
        return sum([dot(g, u) * dsg for g, dsg in zip(g_list, dsg_list)])

    def dlfunc(self, u, du, g_list, dsg_list):
        return sum([dot(g, du) * dsg for g, dsg in zip(g_list, dsg_list)])

    @qtty_to_eval("Disp_Force")
    def Disp_Force(self, phi, U, P):
        u = U[0]
        sub_dom = self.sub[0]
        return sub_dom * sqrt(dot(u, u)) * self.dx

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    @func_to_eval("Displacement")
    def normDisplacement(self, phi, U, P):
        u = U[0]
        return self.chi(phi) * sqrt(dot(u, u))

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
        eps,
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
        self.beta = 2.0
        self.pow_q = 32
        self.trs_st = 50.0
        self.nN = -1
        self.sub = []
        self.ini_linear = False

        self.g_in = [as_vector(_) for _ in g_in]
        self.ds_in = ds_in
        self.g_out = [as_vector(_) for _ in g_out]
        self.ds_out = ds_out

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.F = lambda w: self.Id + grad(w)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, eps)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)

    def lfunc(self, u):

        l_in = sum([dot(g, u) * dsg for g, dsg in zip(self.g_in, self.ds_in)])
        l_out = sum([dot(g, u) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

        return l_in + l_out

    def dlfunc(self, du):

        l_in = sum([dot(g, du) * dsg for g, dsg in zip(self.g_in, self.ds_in)])
        l_out = sum([dot(g, du) * dsg for g, dsg in zip(self.g_out, self.ds_out)])
        return l_in + l_out

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    def desp(self, g, u):
        return (1.0 / sqrt(dot(g, g))) * dot(g, u)

    @qtty_to_eval("InDisp")
    def InDisp(self, phi, U, P):
        u = U[0]
        return sum([self.desp(g, u) * dsg for g, dsg in zip(self.g_in, self.ds_in)])

    @qtty_to_eval("OutDisp")
    def OutDisp(self, phi, U, P):
        u = U[0]
        return sum([self.desp(g, u) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

    @func_to_eval("Displacement")
    def normDisplacement(self, phi, U, P):
        u = U[0]
        return self.chi(phi) * sqrt(dot(u, u))

    @func_to_eval("VonMises")
    def stressVonMises(self, phi, U, P):
        # Von Mises stress calculated using Cauchy stress
        u = U[0]
        sigma = self.em.stress(self.F(u))
        s = sigma - (1.0 / 3.0) * tr(sigma) * self.Id
        return self.chi(phi) * sqrt(1.5 * inner(s, sigma))

    def _W(self, phi, F):
        return self.A(phi) * self.em.W(F)

    def _DW(self, phi, F, dF):
        # First derivative of strain energy density
        return self.A(phi) * inner(dF, self.em.hatQ1(F))

    def _D2W(self, phi, F, dF1, dF2):
        # Second derivative of strain energy density
        return self.A(phi) * inner(dF1, self.em.hatQ2(F, dF2))

    def Phi(self, t):
        return (1.0 + t**self.pow_q) ** (1.0 / self.pow_q) - 1.0

    def DPhi(self, t):
        dphi = (1.0 + t**self.pow_q) ** ((1.0 / self.pow_q) - 1.0)
        return dphi * (t ** (self.pow_q - 1.0))

    def vms_func(self, u):
        sigma = self.em.stress(self.F(u)) / self.trs_st
        Bsigma = 3.0 * sigma - tr(sigma) * self.Id
        return sqrt(0.5 * inner(Bsigma, sigma))

    def _L(self, u):
        sigma = self.em.stress(self.F(u)) / self.trs_st
        Bsigma = 3.0 * sigma - tr(sigma) * self.Id
        vms = sqrt(0.5 * inner(Bsigma, sigma))
        return self.DPhi(vms**2) * Bsigma

    def linear_model(self, phi):
        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        md = Hookecomponents()
        Wf = self.A(phi) * inner(self.F(u), md.hatQ1(grad(v))) * self.dx
        # Robin boundary conditions
        for k, d_out in zip(self.kappa, self.ds_out):
            Wf += k * dot(u, v) * d_out
        # Applied forces (multiplied by a factor)
        for g, d_in in zip(self.g_in, self.ds_in):
            Wf -= dot(g, v) * d_in

        return Wf

    def pde(self, phi):

        if self.em.lineal:
            u = TrialFunction(self.space)
            v = TestFunction(self.space)

            # Weak formulation of state problem
            Wf = self._DW(phi, self.F(u), grad(v)) * self.dx
            # Robin boundary conditions
            for k, d_out in zip(self.kappa, self.ds_out):
                Wf += k * dot(u, v) * d_out
            # Applied forces
            for g, d_in in zip(self.g_in, self.ds_in):
                Wf -= dot(g, v) * d_in

            return [(Wf, self.bc)]

        else:
            u = Coefficient(self.space)
            v = TestFunction(self.space)
            du = TrialFunction(self.space)
            l = Constant(self.domain)

            # Weak formulation of state problem
            Wf = self._DW(phi, self.F(u), grad(v)) * self.dx
            # Robin boundary conditions
            for k, d_out in zip(self.kappa, self.ds_out):
                Wf += k * dot(u, v) * d_out
            # Applied forces
            for g, d_in in zip(self.g_in, self.ds_in):
                Wf -= l * dot(g, v) * d_in

            # Jacobian
            Jc = self._D2W(phi, self.F(u), grad(du), grad(v)) * self.dx
            for k, d_out in zip(self.kappa, self.ds_out):
                Jc += k * dot(du, v) * d_out

            if self.ini_linear:
                ini_par = self.linear_model(phi)
            else:
                ini_par = self.u0

            return [(Wf, self.bc, Jc, u, ini_par, (l, self.nN))]

    def adjoint(self, phi, U):

        p = TrialFunction(self.space)
        q = TestFunction(self.space)
        u = U[0]

        # Weak formulation of adjoint problem
        Wf = self._D2W(phi, self.F(u), grad(p), grad(q)) * self.dx
        # Robin boundary conditions
        for k, d_out in zip(self.kappa, self.ds_out):
            Wf += k * dot(p, q) * d_out
        Wf += self.dlfunc(q)

        # Energy penalization
        # Wf += self.beta * inner(self.em.hatQ1(self.F(u)), grad(q)) * self.dx
        # for k, d_out in zip(self.kappa, self.ds_out):
        #     Wf += self.beta * k * dot(u, q) * d_out

        # Stress penalization
        # Dst = self.em.Dstress(self.F(u), grad(q)) / self.trs_st
        # Wf += self.beta * inner(self._L(u), Dst) * self.dx

        return [(Wf, self.bc)]

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        Jfunc = self.lfunc(u)
        Jfunc += self.alpha * self.chi(phi) * self.dx

        # Energy cost functional
        # Jfunc += self.beta * self.em.W(self.F(u)) * self.dx
        # for k, d_out in zip(self.kappa, self.ds_out):
        #     Jfunc += self.beta * 0.5 * k * dot(u, u) * d_out

        # Stress penalization
        # Jfunc += self.beta * self.Phi(self.vms_func(u) ** 2) * self.dx

        return Jfunc

    def constraint(self, phi, U):
        # There is no constraints
        return []

    def derivative(self, phi, U, P):
        # Shape derivative components
        u = U[0]
        p = P[0]
        Fu = self.F(u)

        S0_J = self.zero_vec

        S1_J = self._DW(phi, Fu, grad(p)) * self.Id
        S1_J += self.alpha * self.chi(phi) * self.Id
        S1_J -= self.A(phi) * grad(p).T * self.em.hatQ1(Fu)
        S1_J -= self.A(phi) * grad(u).T * self.em.hatQ2(Fu, grad(p))

        # Energy derivative components
        # S1_J += self.beta * self.em.W(Fu) * self.Id
        # S1_J -= self.beta * grad(u).T * self.em.hatQ1(Fu)

        # Stress derivative components
        # S1_J += self.beta * self.Phi(self.vms_func(u) ** 2) * self.Id
        # S1_J -= self.beta * grad(u).T * self.em.hatQ3(self._L(u), Fu)
        #

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)
        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class MechanismNeumann(Model):

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
        self.kappa = kappa
        self.alpha = alpha
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
        return self.chi(phi) * sqrt(dot(u, u))

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
        Wf += self.dlfunc(q)

        return [(Wf, self.bc)]

    def cost(self, phi, U):
        u = U[0]
        return self.lfunc(u) + self.alpha * self.chi(phi) * self.dx

    def constraint(self, phi, U):
        return []

    def derivative(self, phi, U, P):
        u = U[0]
        p = P[0]
        F = self.F(u)

        S0_J = self.zero_vec
        Q1 = grad(p).T * self.em.hatQ1(F)
        Q2 = grad(u).T * self.em.hatQ2(F, grad(p))
        S1_J = (self._DW(phi, F, grad(p)) + self.alpha * self.chi(phi)) * self.Id
        S1_J -= self.A(phi) * (Q1 + Q2)

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)
        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class MechanismNeumann2(Model):

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

        l_out = sum([dot(g, u) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

        return l_out

    def dlfunc(self, du):

        l_out = sum([dot(g, du) * dsg for g, dsg in zip(self.g_out, self.ds_out)])

        return l_out

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
        return sqrt(dot(u, u))

    @func_to_eval("VonMises")
    def stressVonMises(self, phi, U, P):
        # Von Mises stress calculated using Cauchy stress
        u = U[0]
        sigma = self.em.stress(self.F(u))
        s = sigma - (1.0 / 3.0) * tr(sigma) * self.Id

        return sqrt(1.5 * inner(s, sigma))

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


class KohnVogelius(Model):

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
        eps,
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
        return self.chi(phi) * sqrt(dot(u, u))

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
            v = TrialFunction(self.space)
            w = TestFunction(self.space)

            Win = self._DW(phi, self.F(u), grad(w)) * self.dx
            for g, dsg in zip(self.g_in, self.ds_in):
                Win -= dot(g, w) * dsg

            Wout = self._DW(phi, self.F(v), grad(w)) * self.dx
            for g, dsg in zip(self.g_out, self.ds_out):
                Wout -= dot(g, w) * dsg

            return [(Win, self.bc), (Wout, self.bc)]

        else:
            u = Coefficient(self.space)
            v = Coefficient(self.space)
            du = TrialFunction(self.space)
            dv = TrialFunction(self.space)
            w = TestFunction(self.space)
            lin = Constant(self.domain)
            lout = Constant(self.domain)
            Fu = self.F(u)
            Fv = self.F(v)

            Win = self._DW(phi, Fu, grad(w)) * self.dx
            for g, dsg in zip(self.g_in, self.ds_in):
                Win -= lin * dot(g, w) * dsg
            Jcin = self._D2W(phi, Fu, grad(du), grad(w)) * self.dx

            Wout = self._DW(phi, Fv, grad(w)) * self.dx
            for g, dsg in zip(self.g_out, self.ds_out):
                Wout -= lout * dot(g, w) * dsg
            Jcout = self._D2W(phi, Fv, grad(dv), grad(w)) * self.dx

            in_case = (Win, self.bc, Jcin, u, self.u0, (lin, self.nN))
            out_case = (Wout, self.bc, Jcout, v, self.u0, (lout, self.nN))
            return [in_case, out_case]

    def adjoint(self, phi, U):

        p = TrialFunction(self.space)
        q = TrialFunction(self.space)
        r = TestFunction(self.space)
        u = U[0]
        v = U[1]
        uv = u - self.kappa * v
        qs = self._DW(phi, self.F(uv), grad(r))
        qs += self._D2W(phi, self.F(uv), grad(uv), grad(r))

        Win = self._D2W(phi, self.F(u), grad(p), grad(r)) * self.dx
        Win += qs * self.dx

        Wout = self._D2W(phi, self.F(v), grad(q), grad(r)) * self.dx
        Wout -= self.kappa * qs * self.dx

        return [(Win, self.bc), (Wout, self.bc)]

    def cost(self, phi, U):
        u = U[0]
        v = U[1]
        uv = u - self.kappa * v
        J = self._DW(phi, self.F(uv), grad(uv)) * self.dx
        J += self.alpha * self.chi(phi) * self.dx
        return J

    def constraint(self, phi, U):
        return []

    def derivative(self, phi, U, P):
        u = U[0]
        v = U[1]
        p = P[0]
        q = P[1]
        uv = u - self.kappa * v

        S0_J = self.zero_vec

        Q1u = grad(p).T * self.em.hatQ1(self.F(u))
        Q2u = grad(u).T * self.em.hatQ2(self.F(u), grad(p))

        Q1v = grad(q).T * self.em.hatQ1(self.F(v))
        Q2v = grad(v).T * self.em.hatQ2(self.F(v), grad(q))

        Quv = self.em.hatQ1(self.F(uv)) + self.em.hatQ2(self.F(uv), grad(uv))

        s1I = self._DW(phi, self.F(uv), grad(uv))
        s1I += self._DW(phi, self.F(u), grad(p))
        s1I += self._DW(phi, self.F(v), grad(q))
        s1I += self.alpha * self.chi(phi)

        S1_J = s1I * self.Id
        S1_J -= grad(uv).T * self.A(phi) * Quv
        S1_J -= self.A(phi) * (Q1u + Q2u)
        S1_J -= self.A(phi) * (Q1v + Q2v)

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)
        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False
