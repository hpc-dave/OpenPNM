import numpy as np
from sympy import symbols, sympify, lambdify
from scipy.sparse import linalg
import scipy.sparse as sprs


__all__ = [
    "init_coefficient_matrix",
    "init_rhs",
    "set_value_bc",
    "set_rate_bc",
    "get_rate",
    "solve",
]


# It is tempting to make these functions accept a dictionary with "conns"
# and to pull row and col data, but the way things are now means absolutely
# no interpretation of the incoming data is required, which is the whole
# point. It is also tempting to return a sparse array, but this might not
# be compatible with gpu's or petsc, so will only convert to that when
# needed.


def to_symmetrical(row, col, data):
    if np.all(row < col) or np.all(row > col):  # am is triu or tril
        row = np.hstack((row, col))
        col = np.hstack((col, row))
        data = np.hstack((data, data))
    return row, col, data


def get_Np(row, col):
    return max(max(row), max(col)) + 1


def get_diags(row, col):
    return np.where(row == col)[0]


def init_coefficient_matrix(row, col, val):
    row, col, data = to_symmetrical(row, col, val)
    Np = get_Np(row, col)
    diag = np.zeros(Np, dtype=float)
    np.add.at(diag, row, -val[col])
    new_row = np.hstack((row, np.arange(Np)))
    new_col = np.hstack((col, np.arange(Np)))
    new_val = np.hstack((val, diag)).T
    return new_row, new_col, new_val


def init_rhs(row, col):
    Np = get_Np(row, col)
    b = np.zeros(Np, dtype=float)
    return b


def set_value_bc(row, col, val, b, values, locs):
    b[locs] = values  # Put value in RHS
    # Zero rows and cols for given locs
    rows = np.isin(row, locs)
    val[rows] = np.zeros(sum(rows == True))
    # Re-add b entries to diagonal of A
    diag = get_diags(row, col)
    val_diag = val[diag]
    val_diag[locs] = np.ones_like(locs)
    val[diag] = val_diag
    return val, b


def set_rate_bc(b, rates, locs):
    b[locs] = -rates
    return b


def get_rate(row, col, val, x, locs):
    Qt = val*(x[row] - x[col]).squeeze()
    Qp = np.zeros_like(x)
    np.add.at(Qp, row, -Qt)
    # np.add.at(Qp, col, -Qt)
    rate = Qp[locs]
    return rate


def to_coo(row, col, val):
    A = sprs.coo_matrix((val, (row, col)))
    return A


def solve(A, b, solver='sp', **kwargs):
    f = globals()["solve_" + solver]
    x = f(A, b, **kwargs)
    return x


def solve_sp(A, b, **kwargs):
    A = sprs.csr_matrix(A)
    x = linalg.spsolve(A=A, b=b, **kwargs)
    return x


# Below here is for reactive transport
def solve_reactive(A, b, x0=None, rxns={}, f=[0, 0], maxiter=1000, rtol=1e-10, atol=1e-5, solver='sp'):
    if x0 is None:
        x0 = np.zeros_like(b)
    b_cached = b.copy()
    A_cached = A.copy()
    i = 0
    while i < maxiter:
        b = b_cached.copy()
        A = A_cached.copy()
        S1, S2, R = eval_source_term(x=x0, **rxns)
        A, b = set_source_term(A=A, b=b, S1=S1, S2=S2, f=f)
        x = solve(A=A, b=b, solver=solver)
        if isconverged(A=A, b=b, x=x, x0=x0, rtol=rtol, atol=atol) and (i > 0):
            res = get_residual(A, b, x)
            print(f'Solution converged after {i} iterations \n',
                  f' Current residual is {abs(res)}')
            break
        else:
            x0 = x
            i += 1
    if i == maxiter:
        res = get_residual(A, b, x)
        print(f'Solution not converged after {i} iterations \n',
              f' Current residual is {res}')
    return A, b, x


def isconverged(A, x, x0=None, b=None, rtol=1e-12, atol=1e-5):
    if not sprs.issparse(x):
        x = sprs.coo_matrix(x).T
    arr_a = (A*x).toarray().squeeze()
    # Ensure solution is constant between iterations such that x=x0
    if x0 is not None:
        if not sprs.issparse(x0):
            x0 = sprs.coo_matrix(x0).T
        arr_a0 = (A*x0).toarray().squeeze()
        flag1 = np.allclose(arr_a, arr_a0, rtol=rtol, atol=atol)
    else:
        flag1 = True
    # Ensure solution is numerically correct such that Ax=b
    if b is not None:
        if not sprs.issparse(b):
            b = sprs.coo_matrix(b).T
        arr_b = (b).toarray().squeeze()
        flag2 = np.allclose(arr_a, arr_b, rtol=rtol, atol=atol)
    else:
        flag2 = True
    return flag1*flag2


def get_residual(A, b, x):
    if not sprs.issparse(x):
        x = sprs.coo_matrix(x).T
    if not sprs.issparse(b):
        b = sprs.coo_matrix(b).T
    res = abs(sprs.linalg.norm(A*x - b))
    return res


def get_source_term(f, **kwargs):
    eqn = sympify(f)
    args = {'x': symbols('x')}
    for key in kwargs.keys():
        args[key] = symbols(key)
    r, s1, s2 = _build_func(eqn, **args)
    return {**{"S1": s1, "S2": s2, "R": r}, **kwargs}


def _build_func(eq, **args):
    eq_prime = eq.diff(args['x'])
    s1 = eq_prime
    s2 = eq - eq_prime*args['x']
    EQ = lambdify(args.values(), expr=eq, modules='numpy')
    S1 = lambdify(args.values(), expr=s1, modules='numpy')
    S2 = lambdify(args.values(), expr=s2, modules='numpy')
    return EQ, S1, S2


def eval_source_term(x, S1, S2, R, locs, **kwargs):
    mask = np.zeros_like(x, dtype=bool)
    mask[locs] = True
    x = np.array(x) + 1e-50
    S1_val = S1(x, **kwargs)
    S2_val = S2(x, **kwargs)
    R_val = R(x, **kwargs)
    S1_val[~mask] = 0.0
    S2_val[~mask] = 0.0
    R_val[~mask] = 0.0
    return S1_val, S2_val, R_val


def set_source_term(A, b, S1, S2, R=None, f=[0, 0]):
    # f = [0, 0] for steady-state
    # f = [0, 1] for explicit
    # f = [1, 1] for implicit
    # f = [0, 2] for crank-nicolson explicit
    # f = [1, 2] for crank-nicolson implicit
    diag = A.diagonal()
    diag += (1-2*f[0])*S1
    A.setdiag(diag)
    b -= (1-2*f[1])*S2
    return A, b


if __name__ == "__main__":
    import openpnm as op
    import matplotlib.pyplot as plt

    pn = op.network.Cubic([5, 5, 1])
    am = pn.create_adjacency_matrix()
    row, col, val = am.row, am.col, am.data
    row, col, val = init_coefficient_matrix(row, col, val)
    b = init_rhs(row, col)
    val, b = set_value_bc(row, col, val, b, values=1.5, locs=[2])
    val, b = set_value_bc(row, col, val, b, values=2.0, locs=[22])
    b = set_rate_bc(b, rates=-2.0, locs=[12])
    A = to_coo(row, col, val)
    x = solve_sp(A, b)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.reshape([5, 5]), origin='lower')
    ax[0].set_title("No reaction")

    func = 'a*(x**b)'
    s = get_source_term(func, a=-1000, b=2)
    s['locs'] = [0]
    A, b, x = solve_reactive(A, b, rxns=s, f=[0, 0], maxiter=100, rtol=1e-12, solver='sp')
    ax[1].imshow(x.reshape([5, 5]), origin='lower')
    ax[1].set_title("With reaction")
