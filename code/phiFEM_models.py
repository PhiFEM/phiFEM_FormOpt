from typing import List
from dolfinx.fem import Function, locate_dofs_topological
from formopt import Model, qtty_to_eval
import numpy as np
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

def _laplacian_direct_dirichlet(space, phi, meas):
    stab_par = 1.0

    domain = space.mesh

    h_T = CellDiameter(domain)
    nv = FacetNormal(domain)
    w = TrialFunction(space)
    phiw = phi * w
    v = TestFunction(space)
    phiv = phi * v

    a = inner(grad(phiw), grad(phiv)) * meas["dx"]((1, 2))
    a -= inner(grad(phiw), nv) * phiv * meas["ds_out"](100)
    Ga = (h_T**2) * div(grad(phiw)) * div(grad(phiv)) * meas["dx"](2)
    Ga += avg(h_T) * jump(grad(phiw), nv) * jump(grad(phiv), nv) * meas["dS"]((2, 3))

    l = phiv * meas["dx"]((1, 2))
    Gl = -(h_T**2) * div(grad(phiv)) * meas["dx"](2)
    return a - l + stab_par * (Ga - Gl)

def _elasticity_neumann(mixed_space, sigma, epsilon, phi, g, meas):
    (u, y, p) = TrialFunctions(mixed_space)
    (v, z, q) = TestFunctions(mixed_space)
    domain = mixed_space.mesh

    hT = CellDiameter(domain)
    nf = FacetNormal(domain)

    yp = dot(y, grad(phi)) + (p * phi / hT)
    zq = dot(z, grad(phi)) + (q * phi / hT)

    gamma_u = 0.01
    gamma_p = 0.01
    sigma_D = 0.1
    gamma_div = 0.1

    su, sv = sigma(u), sigma(v)
    ev = epsilon(v)

    # Weak formulation
    wf = inner(su, ev) * meas["dx"]((1, 2))
    wf += dot(dot(y, nf), v) * meas["ds_out"](100)
    wf += gamma_u * inner(y + su, z + sv) * meas["dx"](2)
    wf += gamma_p * (hT**-2) * dot(yp, zq) * meas["dx"](2)

    # Estabilization
    wf += sigma_D * avg(hT) * inner(jump(su, nf), jump(sv, nf)) * meas["dS"]((2,3))
    wf += gamma_div * inner(div(y), div(z)) * meas["dx"](2)

    # Force application
    wf -= dot(g, v) * meas["dsg"]

    return wf


def _elasticity_interface(mixed_space, sigmas, epsilon, coefs, phi, g, meas):
    u_in, u_out, y_in, y_out, p = TrialFunctions(mixed_space)
    v_in, v_out, z_in, z_out, q = TestFunctions(mixed_space)
    domain = mixed_space.mesh

    cell_diameter = CellDiameter(domain)
    facet_normal = FacetNormal(domain)

    su_in, su_out = sigmas["in"](u_in), sigmas["out"](u_out)
    sv_in, sv_out = sigmas["in"](v_in), sigmas["out"](v_out)
    ev_in, ev_out = epsilon(v_in), epsilon(v_out)

    boundary_in = inner(dot(y_in, facet_normal), v_in)
    boundary_out = inner(dot(y_out, facet_normal), v_out)

    stiffness_in = inner(su_in, ev_in)
    stiffness_out = inner(su_out, ev_out)

    penalization = coefs["penalization"] * (
        inner(y_in + su_in, z_in + sigmas["in"](v_in)) * coefs["out"]
        + inner(y_out + su_out, z_out + sigmas["out"](v_out)) * coefs["in"]
        + cell_diameter ** (-2)
        * inner(
            dot(y_in, grad(phi)) - dot(y_out, grad(phi)),
            dot(z_in, grad(phi)) - dot(z_out, grad(phi)),
        )
        + cell_diameter ** (-2)
        * inner(
            u_in - u_out + cell_diameter ** (-1) * p * phi,
            v_in - v_out + cell_diameter ** (-1) * q * phi,
        )
    )

    stabilization_facets_in = (
        coefs["stabilization"]
        * avg(cell_diameter)
        * inner(jump(su_in, facet_normal), jump(sv_in, facet_normal)))

    stabilization_cells_in = (
        coefs["stabilization"] * cell_diameter**2 * inner(div(y_in), div(z_in))
    )

    stabilization_cells_out = (
        coefs["stabilization"] * cell_diameter**2 * inner(div(y_out), div(z_out))
    )

    stabilization_facets_out = (
        coefs["stabilization"]
        * avg(cell_diameter)
        * inner(jump(su_out, facet_normal), jump(sv_out, facet_normal))
    )

    a = (
        stiffness_in * meas["dx"]((1, 2))
        + stiffness_out * meas["dx"]((2, 3))
        + penalization * meas["dx"](2)
        + stabilization_facets_in * meas["dS"](3)
        + stabilization_facets_out * meas["dS"](4)
        + stabilization_cells_in * meas["dx"](2)
        + stabilization_cells_out * meas["dx"](2)
        + boundary_in * meas["ds_out"](100)
        + boundary_out * meas["ds_out"](101)
    )

    W = a - dot(g, v_in) * meas["dsg"] - dot(g, v_out) * meas["dsg"]

    return W


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


    def postprocess(self, ws: List[Function], ste_fcs: List[Function], phi: Function | None) -> None:
        for w, ste_fc in zip(ws, ste_fcs):
            for i in range(self.rank_dim):
                ste_fc.x.array[i::self.rank_dim] = w.x.array[i::self.rank_dim] * phi.x.array[:]
            ste_fc.x.scatter_forward()

    def pde(self, phi):
        meas = {"dx": self.dx,
                "ds_out": self.ds_out,
                "dS": self.dS}

        wf = _laplacian_direct_dirichlet(self.space, phi, meas)

        return [(wf, [])]

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


class ComplianceVolConstraint(Model):
    def __init__(self, dim, domain, mixed_space, path, rank_dim, g, dsg, bcs, volume):

        self.mixed_space = mixed_space

        self.dim = dim
        self.domain = domain
        self.space, map_in = mixed_space.sub(0).collapse()
        self.maps = [map_in]
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
        self.biform_coefs = (1.0, 1.0)

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    def postprocess(self, ws: List[Function], ste_fcs: List[Function], phi: Function | None) -> None:
        map = self.maps[0]
        for w, ste_fc in zip(ws, ste_fcs):
            ste_fc.x.array[:] = w.x.array[map]

    def pde(self, phi):
        meas = {"dx": self.dx,
                "ds_out": self.ds_out,
                "dS": self.dS,
                "dsg": self.dsg}

        wf = _elasticity_neumann(self.mixed_space, self.sigma, self.epsilon, phi, self.g, meas)

        return [(wf, self.bcs)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        su, eu = self.sigma(u), self.epsilon(u)
        return self.chi(phi) * inner(su, eu) * self.dx

    def constraint(self, phi, U):
        # Volume constraint
        return [(1.0 / self.vol) * self.chi(phi) * self.dx]

    def derivative(self, phi, U, P):
        # Shape derivative components
        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        S0_J = self.zero_vec
        S1_J = self.chi(phi) * (2.0 * grad(u).T * su - inner(su, eu) * self.Id)

        # volume constraint
        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        return (S0_J, [S0_C]), (S1_J, [S1_C])

    def bilinear_form(self, th, xi):
        # Weighted H1-norm with penalty terms
        c0, c1 = self.biform_coefs
        nv = FacetNormal(self.domain)

        B = c0 * dot(th, xi) * self.dx
        B += c1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False

class InterfaceComplianceVolConstraint(Model):
    def __init__(self, dim, domain, mixed_space, path, rank_dim, g, dsg, bcs, volume):

        self.mixed_space = mixed_space

        self.dim = dim
        self.domain = domain
        self.space, map_in = mixed_space.sub(0).collapse()
        map_out = mixed_space.sub(1).collapse()[1]
        self.maps = [map_in, map_out]
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

        self.coef_out = 1.e-4
        self.sigmas = {"in": self.sigma,
                       "out": lambda v: self.coef_out * self.sigma(v)}
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        self.vol = volume
        self.biform_coefs = (1.0, 1.0)

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    def postprocess(self, ws: List[Function], ste_fcs: List[Function], phi: Function | None) -> None:
        map = self.maps[0]
        for w, ste_fc in zip(ws, ste_fcs):
            ste_fc.x.array[:] = w.x.array[map]

    def pde(self, phi):
        meas = {"dx": self.dx,
                "ds_out": self.ds_out,
                "dS": self.dS,
                "dsg": self.dsg}

        coefs = {"penalization": 1.0,
                 "in": self.coef_out/(self.coef_out + 1.0),
                 "out": 1./(self.coef_out + 1.0),
                 "stabilization": 1.0}
        wf = _elasticity_interface(self.mixed_space, self.sigmas, self.epsilon, coefs, phi, self.g, meas)

        return [(wf, self.bcs)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        su, eu = self.sigma(u), self.epsilon(u)
        return self.chi(phi) * inner(su, eu) * self.dx

    def constraint(self, phi, U):
        # Volume constraint
        return [(1.0 / self.vol) * self.chi(phi) * self.dx]

    def derivative(self, phi, U, P):
        # Shape derivative components
        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        S0_J = self.zero_vec
        S1_J = self.chi(phi) * (2.0 * grad(u).T * su - inner(su, eu) * self.Id)

        # volume constraint
        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        return (S0_J, [S0_C]), (S1_J, [S1_C])

    def bilinear_form(self, th, xi):
        # Weighted H1-norm with penalty terms
        c0, c1 = self.biform_coefs
        nv = FacetNormal(self.domain)

        B = c0 * dot(th, xi) * self.dx
        B += c1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class ComplianceVolPenalty(Model):
    def __init__(self, dim, domain, mixed_space, path, rank_dim, g, dsg, bcs, alpha):

        self.mixed_space = mixed_space

        self.dim = dim
        self.domain = domain
        self.space, map_in = mixed_space.sub(0).collapse()
        self.maps = [map_in]
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
        self.alpha = alpha
        self.biform_coefs = (1.0, 1.0)

    @qtty_to_eval("Volume")
    def Volume(self, phi, U, P):
        return self.chi(phi) * self.dx

    def postprocess(self, ws: List[Function], ste_fcs: List[Function], phi: Function | None) -> None:
        map = self.maps[0]
        for w, ste_fc in zip(ws, ste_fcs):
            ste_fc.x.array[:] = w.x.array[map]

    def pde(self, phi):
        meas = {"dx": self.dx,
                "ds_out": self.ds_out,
                "dS": self.dS,
                "dsg": self.dsg}

        wf = _elasticity_neumann(self.mixed_space, self.sigma, self.epsilon, phi, self.g, meas)

        return [(wf, self.bcs)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):
        # Cost functional
        u = U[0]
        su, eu = self.sigma(u), self.epsilon(u)
        J = self.chi(phi) * inner(su, eu) * self.dx
        J += self.alpha * self.chi(phi) * self.dx
        return J

    def constraint(self, phi, U):
        # Volume constraint
        return []

    def derivative(self, phi, U, P):
        # Shape derivative components
        u = U[0]
        su, eu = self.sigma(u), self.epsilon(u)

        S0_J = self.zero_vec
        S1_J = self.chi(phi) * (2.0 * grad(u).T * su - inner(su, eu) * self.Id)
        S1_J += self.alpha * self.chi(phi) * self.Id

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):
        # Weighted H1-norm with penalty terms
        c0, c1 = self.biform_coefs
        nv = FacetNormal(self.domain)

        B = c0 * dot(th, xi) * self.dx
        B += c1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False
