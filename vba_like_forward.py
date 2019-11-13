# %%
import numpy as np
import torch

TR = 2.0
n_t = round(1e2/TR)
dtU = round(2 / TR) + 1
t0U = round(10/TR) + 1
microDT = 2e-1
homogenous = 0
reduced_f = 0
lin = 1
stochastic = 0
alpha = np.inf
sigma = 1e0
nconfounds = 0

# %%
u = np.zeros((2, n_t))
u[0, t0U: t0U + dtU] = 1
u[0, 4 * t0U: 4 * t0U + dtU] = 1
u[1, 4 * t0U: 4 * t0U + 5 * dtU] = 1
nu = u.shape[0]

# %%
A = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
nreg = A.shape[0]
B = np.zeros((2, nreg, nreg)).astype('int')
B[1] = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
C = np.array([[1, 0], [0, 0], [0, 0]])

D = np.zeros((nreg, nreg, nreg)).astype('int')
# %%


class DCM:

    def __init__(self, A, B, C, D, TR=2.0, microDT=0.1, homogenous=1,
                 linearized=0, logx2=1, TE=0.04):
        # Initialization of the evolution part of a standard DCM
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.TR = TR
        self.linearized = linearized
        self.logx2 = logx2
        self.TE = TE
        self.microDT = microDT
        self.decim = np.max([1, np.ceil(TR/microDT)])
        self.nreg = A.shape[0]
        self.nu = C.shape[1]

        self.homogenous = homogenous
        evolution = {}
        observation = {}
        dimensions = {}
        evolution['A'] = A
        evolution['B'] = B
        evolution['C'] = C
        evolution['D'] = D
        evolution['indA'] = np.arange(np.sum(A))
        evolution['indB'] = np.arange(np.sum(B)) + len(evolution['indA'])
        evolution['indC'] = (np.arange(np.sum(C)) + len(evolution['indA'])
                             + len(evolution['indB']))
        evolution['indD'] = (np.arange(np.sum(D)) + len(evolution['indA'])
                             + len(evolution['indB']) + len(evolution['indC']))
        evolution_offset = (len(evolution['indA']) + len(evolution['indB'])
                            + len(evolution['indC']) + len(evolution['indD']))

        evolution['indself'] = np.arange(1) + evolution_offset
        # Initialize hemodynamic parameter indices
        evolution['n1'] = np.arange(0, 5 * self.nreg, 5)
        evolution['n2'] = np.arange(1, 5 * self.nreg, 5)
        evolution['n3'] = np.arange(2, 5 * self.nreg, 5)
        evolution['n4'] = np.arange(3, 5 * self.nreg, 5)
        evolution['n5'] = np.arange(4, 5 * self.nreg, 5)

        evolution['ind1'] = evolution['n1'] + evolution_offset + 1
        evolution['ind2'] = evolution['n2'] + evolution_offset + 1
        evolution['ind3'] = evolution['n3'] + evolution_offset + 1
        evolution['ind4'] = evolution['n4'] + evolution_offset + 1
        evolution['ind5'] = evolution['n5'] + evolution_offset + 1
        evolution['confounds_indu'] = np.arange(0, self.nu)
        evolution['xshift'] = 0
        evolution['deltat'] = self.TR / self.decim

        self.evolution = evolution
        # initialize observation function
        observation['n1'] = evolution['n1']
        observation['n2'] = evolution['n2']
        observation['n3'] = evolution['n3']
        observation['n4'] = evolution['n4']
        observation['n5'] = evolution['n5']

        if homogenous:
            observation['ind1'] = np.arange(1)
            observation['ind2'] = np.arange(1) + 1
        else:
            observation['ind1'] = np.arange(0, 2 * self.nreg, 2)
            observation['ind2'] = np.arange(1, 2 * self.nreg, 2)

        observation['confounds_indu'] = np.arange(0, self.nu)
        self.observation = observation

        dimensions['theta'] = evolution['ind5'][-1] + 1
        dimensions['phi'] = observation['ind2'][-1] + 1
        dimensions['p'] = self.nreg  # Number of regions?
        dimensions['n'] = 5 * self.nreg  # Number of states
        self.dimensions = dimensions

    def parameters_to_matrix(self, theta):
        evolution = self.evolution
        A = evolution['A'].copy()
        A[A != 0] = theta[evolution['indA'], 0]
        A = A - np.exp(theta[evolution['indself']]) * np.eye(A.shape[0])

        B = evolution['B'].copy()
        B[B != 0] = theta[evolution['indB'], 0]

        C = evolution['C'].copy()
        C[C != 0] = theta[evolution['indC'], 0]

        D = evolution['D'].copy()
        D[D != 0] = theta[evolution['indD'], 0]

        return A, B, C, D
# %%


def f_dcm_w_hrf(Xt, theta, ut, DCM):
    evolution = DCM.evolution
    ut = ut[evolution['confounds_indu']]
    xn = Xt[evolution['n5']]
    fx = f_hrf(Xt, theta, xn, DCM)
    fxn = f_dcm_fmri(xn, theta, ut, DCM)
    fx[evolution['n5']] = fxn
    return fx


def f_hrf(Xt, theta, ut, DCM):
    evolution = DCM.evolution
    hrf_epsilon = 1
    hrf_alpha = 0.32 * np.exp(theta[evolution['ind5']])
    hrf_e0 = 1 / (1 + np.exp(-1 * (theta[evolution['ind1']] - 0.6633)))
    hrf_tau0 = 2 * np.exp(theta[evolution['ind2']])
    hrf_kaf = 0.41 * np.exp(theta[evolution['ind3']])
    hrf_kas = 0.65 * np.exp(theta[evolution['ind4']])

    # Linearized option not yet implemented
    # x1 = np.zeros((DCM.nreg, 1))
    # x2 = np.ones((DCM.nreg, 1))
    # x3 = np.ones((DCM.nreg, 1))
    # x4 = np.ones((DCM.nreg, 1))

    x1 = Xt[evolution['n1'], :]

    if not DCM.logx2:
        x2 = Xt[evolution['n2'], :] + 1
    else:
        x2 = np.exp(Xt[evolution['n2'], :]) + evolution['xshift']

        x3 = np.exp(Xt[evolution['n3'], :])
        x4 = np.exp(Xt[evolution['n4'], :])

    fv = x3 ** (1 / hrf_alpha)
    ff = (1 - (1 - hrf_e0) ** (1 / x2)) / hrf_e0

    f = np.zeros((Xt.shape[0], 1))
    print(f.shape)
    print(f[evolution['n1']])
    print(ut.shape)
    print(hrf_kas.shape)
    print(x1.shape)
    print(x2.shape)

    f[evolution['n1']] = hrf_epsilon * ut - hrf_kas * x1 - hrf_kaf * (x2 - 1)
    if not DCM.logx2:
        f[evolution['n2']] = x1
    else:
        f[evolution['n2']] = x1 / x2
    f[evolution['n3']] = (x2 - fv) / (hrf_tau0 * x3)
    f[evolution['n4']] = (x2 * ff / x4 - fv / x3) / hrf_tau0

    fx = Xt + evolution['deltat'] * f

    return fx


def f_dcm_fmri(Xt, theta, ut, DCM):
    A, B, C, D = DCM.parameters_to_matrix(theta)
    B = ut[np.newaxis, :] @ (B)
    D = Xt @ D
    C = C @ ut[:, np.newaxis]

    flow = A + B + D
    xt = flow.dot(Xt) + C

    return xt


# %%
dcm = DCM(A, B, C, D, TR, microDT)

Xt = np.zeros((dcm.dimensions['n'], n_t))
theta = np.random.rand(dcm.dimensions['theta'])
theta = theta[:, np.newaxis]
# %%
for n, ut in enumerate(u.T):
    if n == 0:
        Xtprev = np.zeros((dcm.dimensions['n'], 1))
    else:
        Xtprev = Xt[:, n-1]
    Xt[:, n] = f_dcm_w_hrf(Xtprev, theta, ut, dcm)

# %%
