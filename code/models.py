from distributed import Model

from ufl import (
    TrialFunction, TestFunction,
    FacetNormal, Identity, Measure,
    SpatialCoordinate, Coefficient,
    conditional, indices, as_vector,
    inner, outer, grad, sym, dot,
    lt, pi, cos, sqrt, nabla_div
)

"""
Models:
    Compliance
    CompliancePlus
    InverseElasticity
    Heat
    HeatPlus
    HeatMultiple
    Logistic
"""

class Compliance(Model):
    """
    Models the compliance minimization
    problem for linear elasticity.
    """

    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, vol, path):
        
        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        
        self.dx = Measure("dx", domain = domain)
        self.ds = Measure("ds", domain = domain)		
        self.g = as_vector(g)
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []

        E, nu = 1.0, 0.3
        lmbda = E*nu/(1.0 + nu)/(1.0 - 2.0*nu)
        mu = E/2.0/(1.0 + nu)

        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda w: sym(grad(w))
        self.sigma = lambda w: (
            lmbda*nabla_div(w)*self.Id +
            2.0*mu*self.epsilon(w)
        )
        self.A = lambda w: (
            conditional(lt(w, 0.0), 1.0, 1e-4)
        )
        self.chi = lambda w: (
            conditional(lt(w, 0.0), 1.0, 0.0)
        )
        
    def pde(self, phi):
        
        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        su = self.sigma(u)
        ev = self.epsilon(v)

        W = self.A(phi)*inner(su, ev)*self.dx
        W -= dot(self.g, v)*self.ds_g
        
        return [(W, self.bc)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):
        
        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        J = self.A(phi)*(inner(su, eu))*self.dx
        
        return J

    def constraint(self, phi, U):
        
        C = (1.0/self.vol)*self.chi(phi)*self.dx
        
        return [C]

    def derivative(self, phi, U, P):
        
        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        S0_J = self.zero_vec
        S1_J = 2.0*grad(u).T*su 
        S1_J -= inner(su, eu)*self.Id
        S1_J *= self.A(phi)
            
        S0_C = self.zero_vec
        S1_C = (1.0/self.vol)*self.chi(phi)*self.Id
        
        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])
        
        return S0, S1
    
    def bilinear_form(self, th, xi):
        
        nv = FacetNormal(self.domain)
        
        B = 0.1*dot(th, xi)*self.dx
        B += inner(grad(th), grad(xi))*self.dx
        B += 1e4*dot(th, nv)*dot(xi, nv)*self.ds
        for sb in self.sub:
            B += 1e5*sb*dot(th, xi)*self.dx
        
        return B, False


class CompliancePlus(Model):
    """
    Models a compliance optimization
    problem with multiple load cases
    """
    
    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, vol, path):
        
        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        
        self.dx = Measure("dx", domain = domain)
        self.ds = Measure("ds", domain = domain)
        self.g = [as_vector(_) for _ in g]
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []
        
        E, nu = 1.0, 0.3
        lmbda = E*nu/(1.0 + nu)/(1.0 - 2.0*nu)
        mu = E/2.0/(1.0 + nu)

        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda v: sym(grad(v))
        self.sigma = lambda v: (
            lmbda*nabla_div(v)*self.Id + 
            2.0*mu*self.epsilon(v)
        )
        self.A = lambda v: (
            conditional(lt(v, 0.0), 1.0, 1e-4)
        )
        self.chi = lambda w: (
            conditional(lt(w, 0.0), 1.0, 0.0)
        )
        
    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        su = self.sigma(u)
        ev = self.epsilon(v)

        pdes = []
        for g, ds_g in zip(self.g, self.ds_g):
            W = self.A(phi)*inner(su, ev)*self.dx
            W -= dot(g, v)*ds_g
            pdes.append((W, self.bc))
        
        return pdes

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):

        J = []
        for u in U:	
            su = self.sigma(u)
            eu = self.epsilon(u)
            J.append(
                self.A(phi)*inner(su, eu)*self.dx
            )
        
        return sum(J)

    def constraint(self, phi, U):
        
        C = (1.0/self.vol)*self.chi(phi)*self.dx
        
        return [C]

    def derivative(self, phi, U, P):
        
        S0_J = self.zero_vec

        S1_J = []
        for u in U:
            su = self.sigma(u)
            eu = self.epsilon(u)
            s1 = 2.0*grad(u).T*su 
            s1 -= inner(su, eu)*self.Id
            S1_J.append(self.A(phi)*s1)

        S0_C = self.zero_vec
        S1_C = (1.0/self.vol)*self.chi(phi)*self.Id

        S0 = (S0_J, [S0_C])
        S1 = (sum(S1_J), [S1_C])
        
        return S0, S1
    
    def bilinear_form(self, th, xi):
        
        nv = FacetNormal(self.domain)
        
        B = 0.1*dot(th, xi)*self.dx
        B += inner(grad(th), grad(xi))*self.dx
        B += 1e4*dot(th, nv)*dot(xi, nv)*self.ds
        for sb in self.sub:
            B += 1e4*sb*dot(th, xi)*self.dx

        return B, False


class InverseElasticity(Model):

    def __init__(self, dim, domain, space, forces, ds_forces, ds1, dirbc_partial, dirbc_total, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        
        self.dx = Measure("dx", domain = domain)
        self.fs = [as_vector(f) for f in forces]
        self.dfs = ds_forces
        self.bcF = dirbc_partial
        self.bcG = dirbc_total
        self.gs = []
        self.ds1 = ds1
        self.N = len(self.fs)

        E, nu = 1.0, 0.3
        lmbda = E*nu/(1.0 + nu)/(1.0 - 2.0*nu)
        mu = E/2.0/(1.0 + nu)
        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda w: sym(grad(w))
        self.sigma = lambda w: lmbda*nabla_div(w)*self.Id + 2.0*mu*self.epsilon(w)
        self.A = lambda w: conditional(lt(w, 0.0), 10.0, 1.0)
        self.alpha = 1.0
        self.beta = 1.0
        self.phi = None

    def f_prob(self, u, w, phi, f, df):
        # f-problem

        su = self.sigma(u)
        ew = self.epsilon(w)

        W = self.A(phi)*inner(su, ew)*self.dx
        W -= dot(f, w)*df
        
        return [W, self.bcF]
    
    def g_prob(self, v, w, phi, g):
        # g-problem

        sv = self.sigma(v)
        sg = self.sigma(g)
        ew = self.epsilon(w)

        W = self.A(phi)*inner(sv, ew)*self.dx
        W += self.A(phi)*inner(sg, ew)*self.dx
        
        return [W, self.bcG]
    
    def adj_f_prob(self, p, r, phi, u, v, g):
        # adjoint of f-problem
        
        sp = self.sigma(p)
        er = self.epsilon(r)
        a_g = u - v - g
        b_g = u - g

        W = self.A(phi)*inner(sp, er)*self.dx
        W += self.alpha*dot(a_g, r)*self.dx
        W += self.beta*dot(b_g, r)*self.ds1
        
        return (W, self.bcF)

    def adj_g_prob(self, q, r, phi, u, v, g):
        # adjoint of g-problem

        sq = self.sigma(q)
        er = self.epsilon(r)
        a_g = u - v - g

        W = self.A(phi)*inner(sq, er)*self.dx
        W -= self.alpha*dot(a_g, r)*self.dx
        
        return (W, self.bcG)
    
    def pde0(self, phi):

        a = TrialFunction(self.space)
        b = TestFunction(self.space)

        F = [self.f_prob(a, b, phi, f, df) for f, df in zip(self.fs, self.dfs)]

        return F
    
    def pde(self, phi):

        a = TrialFunction(self.space)
        b = TestFunction(self.space)

        F = [self.f_prob(a, b, phi, f, df) for f, df in zip(self.fs, self.dfs)]
        G = [self.g_prob(a, b, phi, g) for g in self.gs]
        
        return F + G

    def adjoint(self, phi, U):
        
        a = TrialFunction(self.space)
        b = TestFunction(self.space)

        AF = [self.adj_f_prob(a, b, phi, u, v, g) for u, v, g in zip(U[:self.N], U[self.N:], self.gs)]
        AG = [self.adj_g_prob(a, b, phi, u, v, g) for u, v, g in zip(U[:self.N], U[self.N:], self.gs)]
        
        return AF + AG
    
    def cost(self, phi, U):
        
        uvg = zip(U[:self.N], U[self.N:], self.gs)
        ug = zip(U[:self.N], self.gs)

        Ja = [dot(u - v - g, u - v - g)*self.dx for u, v, g in uvg]

        Jb = [dot(u - g, u - g)*self.ds1 for u, g in ug]
        
        return (self.alpha/2.0)*sum(Ja) + (self.beta/2.0)*sum(Jb)

    def constraint(self, phi, U):
        return []
    
    def S0(self, u, v, q, g, phi):
        
        i, j, k = indices(3)
        sq = self.sigma(q)
        eg = self.epsilon(g)
        
        s0i = grad(g).T*(u - v - g)
        s0j = as_vector(sq[i, j]*(grad(eg))[i, j, k], (k))
        
        return -self.alpha*s0i + self.A(phi)*s0j

    def S1(self, u, v, p, q, g, phi):

        su = self.sigma(u)
        sp = self.sigma(p)
        sq = self.sigma(q)
        svg = self.sigma(v + g)
        ep = self.epsilon(p)
        eq = self.epsilon(q)
        uvg = u - v - g

        s1i = inner(su, ep) + inner(svg, eq)
        s1i = self.alpha*dot(uvg, uvg)/2.0 + self.A(phi)*s1i

        s1j = grad(u).T*sp + grad(p).T*su 
        s1j += grad(v).T*sq + grad(q).T*svg

        return s1i*self.Id - self.A(phi)*s1j

    def derivative(self, phi, U, P):
        
        fu = U[:self.N]
        gv = U[self.N:]
        fp = P[:self.N]
        gq = P[self.N:]

        uvqg = zip(fu, gv, gq, self.gs)
        uvpqg = zip(fu, gv, fp, gq, self.gs)

        S0 = [self.S0(u, v, q, g, phi) for u, v, q, g in uvqg]
        S1 = [self.S1(u, v, p, q, g, phi) for u, v, p, q, g in uvpqg]

        return (sum(S0), []), (sum(S1), [])
    
    def bilinear_form(self, th, xi):
        biform = 0.1*dot(th, xi)*self.dx + inner(grad(th), grad(xi))*self.dx
        return biform, True


class Heat(Model):
    
    def __init__(self, dim, domain, space, dir_bcs, vol, path, sc_type = "Uniform"):
        
        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain = domain)
        self.ds = Measure("ds", domain = domain)
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []
        
        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, 1e-2)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        
        self.sc_type = sc_type
        self.f = self.source()
    
    def source(self):
        if self.sc_type == "Uniform":
            return 1.0
        elif self.sc_type == "1Load":
            x0, y0 = 0.5, 0.5
            max_value, epsilon = 50.0,  0.1
            x = SpatialCoordinate(self.domain)
            r = sqrt((x[0] - x0)**2 + (x[1] - y0)**2)
            delta_expr = conditional(
                lt(r, epsilon),
                max_value*(1.0 + cos(pi*r/epsilon))/2.0,
                0.0
            )
            return delta_expr

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)

        W = self.A(phi)*dot(grad(u), grad(v))*self.dx
        W -= self.f*v*self.dx
        
        return [(W, self.bc)]
    
    def adjoint(self, phi, U):
        return []
    
    def cost(self, phi, U):
        
        u = U[0]
        J = self.f*u*self.dx
        
        return J
    
    def constraint(self, phi, U):
        
        C = (1.0/self.vol)*self.chi(phi)*self.dx
        
        return [C]
    
    def derivative(self, phi, U, P):
        
        u = U[0]

        if self.sc_type == "Uniform":
            S0_J = self.zero_vec
        elif self.sc_type == "1Load":
            S0_J = 2.0*u*grad(self.f)
        
        S1_J = (2.0*u*self.f - self.A(phi)*dot(grad(u), grad(u)))*self.Id
        S1_J += 2.0*self.A(phi)*outer(grad(u), grad(u))
        
        S0_C = self.zero_vec
        S1_C = (1.0/self.vol)*self.chi(phi)*self.Id
        
        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])
        
        return S0, S1
    
    def bilinear_form(self, th, xi):
        
        B = inner(grad(th), grad(xi))*self.dx
        for sb in self.sub:
            B += 1e4*sb*dot(th, xi)*self.dx
        
        bc = True
        
        if self.sc_type == "1Load":
            nv = FacetNormal(self.domain)
            B += 1e4*dot(th, nv)*dot(xi, nv)*self.ds
            bc = False
        
        return B, bc


class HeatPlus(Model):

    def __init__(self, dim, domain, space, dir_bcs, vol, path):
        
        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain = domain)
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []
        self.wts = [1.0 for _ in range(len(dir_bcs))]
        
        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)

    @staticmethod
    def A(w):
        return conditional(lt(w, 0.0), 1.0, 1e-2)
    
    @staticmethod
    def chi(w):
        return conditional(lt(w, 0.0), 1.0, 0.0)

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)

        W = self.A(phi)*dot(grad(u), grad(v))*self.dx - v*self.dx
        
        return [[W, [bc]] for bc in self.bc]
    
    def adjoint(self, phi, U):
        return []
    
    def cost(self, phi, U):
        
        J = [wt*u*self.dx for wt, u in zip(self.wts, U)]
        
        return sum(J)
    
    def constraint(self, phi, U):
        
        C = (1.0/self.vol)*self.chi(phi)*self.dx
        
        return [C]

    def derivative(self, phi, U, P):
        
        u = U[0]

        S0_J = self.zero_vec
        
        S1_J = []
        for wt, u in zip(self.wts, U):
            s1 = (2.0*u - self.A(phi)*dot(grad(u), grad(u)))*self.Id
            s1 += 2.0*self.A(phi)*outer(grad(u), grad(u))
            S1_J.append(wt*s1)
        
        S0_C = self.zero_vec
        S1_C = (1.0/self.vol)*self.chi(phi)*self.Id
        
        S0 = [S0_J, [S0_C]]
        S1 = [sum(S1_J), [S1_C]]
        
        return S0, S1
    
    def bilinear_form(self, th, xi):

        B = inner(grad(th), grad(xi))*self.dx
        for sb in self.sub:
            B += 1e4*sb*dot(th, xi)*self.dx
            
        return B, True


class HeatMultiple(Model):

    def __init__(self, dim, domain, space, dirichlet_bcs, vol):

        self.dim = dim
        self.domain = domain
        self.space = space

        self.dx = Measure("dx", domain = domain)
        self.ds = Measure("ds", domain = domain)
        self.bcs = dirichlet_bcs
        self.vol = vol
        self.wts = None
        self.gs = None
        self.fs = None
        self.subs = None

        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)
        self.A = lambda w: conditional(lt(w, 0.0), 1.0, 1e-3)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)

    def state_prob(self, u, v, phi, g, bc):
        
        W = self.A(phi)*dot(grad(u), grad(v))*self.dx
        W += self.A(phi)*dot(grad(g), grad(v))*self.dx
        
        return [W, bc]
    
    def adjoint_prob(self, p, q, phi, wt, u, g, bc):
        
        W = self.A(phi)*dot(grad(p), grad(q))*self.dx
        W += 2.0*wt*dot(grad(u + g), grad(q))*self.dx

        return [W, bc]

    def pde(self, phi):
        
        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        
        S = [
            self.state_prob(u, v, phi, g, bc)
            for g, bc in zip(self.gs, self.bcs)
        ]

        return S
    
    def adjoint(self, phi, U):
        
        p = TrialFunction(self.space)
        q = TestFunction(self.space)

        A = [
            self.adjoint_prob(p, q, phi, wt, u, g, bc)
            for wt, u, g, bc in zip(self.wts, U, self.gs, self.bcs)
        ]

        return A

    def cost(self, phi, U):
        
        J = [
            wt*dot(grad(u + g), grad(u + g))*self.dx
            for wt, u, g in zip(self.wts, U, self.gs)
        ]

        return sum(J)

    def constraint(self, phi, U):
        
        C = self.chi(phi)/self.vol*self.dx
        
        return [C]

    def S0(self, phi, wt, u, g, p):

        D2g = grad(grad(g))

        s0 = 2.0*wt*D2g*grad(u + g)
        s0 += self.A(phi)*D2g*grad(p)
        
        return s0

    def S1(self, phi, wt, u, g, p):
        
        grad_ug = grad(u + g)

        s1 = wt*dot(grad_ug, grad_ug)*self.Id
        s1 -= 2.0*wt*outer(grad(u), grad_ug)
        s1 += self.A(phi)*dot(grad(p), grad_ug)*self.Id
        s1 -= self.A(phi)*outer(grad(p), grad_ug)
        s1 -= self.A(phi)*outer(grad(u), grad(p))
        
        return s1

    def derivative(self, phi, U, P):
        
        S0_J = []
        S1_J = []

        for wt, u, g, p in zip(self.wts, U, self.gs, P):
            S0_J.append(self.S0(phi, wt, u, g, p))
            S1_J.append(self.S1(phi, wt, u, g, p))
    
        S0_C = self.zero_vec
        S1_C = self.chi(phi)/self.vol*self.Id

        S0 = [sum(S0_J), [S0_C]]
        S1 = [sum(S1_J), [S1_C]]
    
        return S0, S1

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)
        
        B = inner(grad(th), grad(xi))*self.dx
        B += 1e4*dot(th, nv)*dot(xi, nv)*self.ds
        
        for sb in self.subs:
            B += 1e5*sb*dot(th, xi)*self.dx
        
        return B, False


class Logistic(Model):

    def __init__(self, dim, domain, space, r, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path
        
        self.dx = Measure('dx', domain = domain)
        self.ds = Measure("ds", domain = domain)
        
        self.r = r
        self.vol = 0.5
        self.ini_func = None

        self.zero_vec = as_vector(dim*[0.0])
        self.Id = Identity(dim)
        self.K = lambda w: (
            conditional(lt(w, 0.0), 1.0, 0.01)
        )
        self.chi = lambda w: (
            conditional(lt(w, 0.0), 1.0, 0.0)
        )

    def pde(self, phi):
        
        u = Coefficient(self.space)
        v = TestFunction(self.space)
        du = TrialFunction(self.space)

        F = dot(grad(u), grad(v))*self.dx
        F -= self.r*u*(1.0 - u/self.K(phi))*v*self.dx

        J = dot(grad(du), grad(v))*self.dx
        J -= self.r*du*(1.0 - 2.0*u/self.K(phi))*v*self.dx

        return [(F, [], J, u, self.ini_func)]

    def adjoint(self, phi, U):
        
        u = U[0]

        p = TrialFunction(self.space)
        q = TestFunction(self.space)
        
        W = dot(grad(p), grad(q))*self.dx
        W += self.r*p*q*(2.0*u/self.K(phi) - 1.0)*self.dx
        W -= q*self.dx 

        return [(W, [])]

    def cost(self, phi, U):
        u = U[0]
        J = -u*self.dx
        return J

    def constraint(self, phi, U):
        
        C = (1.0/self.vol)*self.chi(phi)*self.dx
        
        return [C]

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]

        S0_J = self.zero_vec
        S1_J = (-u + dot(grad(u), grad(p)))*self.Id 
        S1_J -= self.r*u*(1.0 - u/self.K(phi))*p*self.Id
        S1_J -= (outer(grad(u), grad(p)) + outer(grad(p), grad(u)))
            
        S0_C = self.zero_vec
        S1_C = (1.0/self.vol)*self.chi(phi)*self.Id
        
        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])
        
        return S0, S1

    def bilinear_form(self, th, xi):
        
        nv = FacetNormal(self.domain)
        
        B = 0.1*dot(th, xi)*self.dx
        B += inner(grad(th), grad(xi))*self.dx
        B += 1e4*dot(th, nv)*dot(xi, nv)*self.ds
        
        return B, False

