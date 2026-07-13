import math

from kd_legacy import kdcalc_fortran

# 系统成分 (data_original case 1)
cs = [0.0, 0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28]

# 化学计量 (1-based)
uaj = [[0.0]*8 for _ in range(4)]
uaj[1][1] = 1.666667
uaj[1][2] = 2.5
uaj[2][3] = 1.0
uaj[2][4] = 1.0
uaj[3][3] = 2.0
uaj[3][4] = 2.0
uaj[3][5] = 1.0
uaj[3][1] = 1.333333
uaj[3][2] = 2.0
uaj[3][6] = 1.0

ta = [0.0, 1.0, 0.666667, 1.0]


def kdcalc(tk, clj=None):
    # Original debug_step2 used inline 10**(A/T + B) (= modern form)
    return kdcalc_fortran(tk, kdmode=3, clj_1based=clj, form="modern")


def compute_q(fa, kd):
    fl = max(1e-14, 1.0 - sum(fa[1:4]))
    rj = [0.0]*8
    clj = [0.0]*8
    caj = [[0.0]*8 for _ in range(4)]
    for j in range(1, 7):
        denom = 1.0
        for l in range(1, 4):
            denom += fa[l] * (kd[l][j] - 1.0)
        rj[j] = 1.0 / denom
        clj[j] = cs[j] * rj[j]
        for l in range(1, 4):
            caj[l][j] = kd[l][j] * clj[j]
    q = [0.0]*4
    for l in range(1, 4):
        q[l] = -ta[l]
        for j in range(1, 7):
            q[l] += uaj[l][j] * caj[l][j]
    return q, clj, caj, rj, fl

def solve_linear(A, b):
    n = len(A)
    # Gaussian elimination with partial pivoting
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        max_row = k
        for i in range(k+1, n):
            if abs(M[i][k]) > abs(M[max_row][k]):
                max_row = i
        M[k], M[max_row] = M[max_row], M[k]
        if abs(M[k][k]) < 1e-14:
            return None
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = M[i][n]
        for j in range(i+1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

# T=1668.16, 1atm
tk = 1668.16
kd = kdcalc(tk)
fa = [0.0]*4
fa[1] = 0.693
fa[2] = 0.198

for it in range(50):
    q, clj, caj, rj, fl = compute_q(fa, kd)
    print(f"iter {it}: fl={fl:.6e} fa1={fa[1]:.6f} fa2={fa[2]:.6f} fa3={fa[3]:.6f} q1={q[1]:.6f} q2={q[2]:.6f} q3={q[3]:.6f}")
    active = [l for l in [1,2,3] if q[l] <= 0]
    print(f"  active: {active}")
    if len(active) == 0:
        print("  converged: no active phases")
        break
    if len(active) == 1:
        l = active[0]
        print(f"  single phase {l}, bisecting")
        f_lo, f_hi = 0.0, 0.9999
        q_lo = compute_q([0.0, fa[1] if l==1 else 0.0, fa[2] if l==2 else 0.0, fa[3] if l==3 else 0.0], kd)[0][l]
        # q_lo needs current fa for other phases which are zero now
        fa_test = [0.0]*4
        q_at_f = {}
        for f in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 0.9999]:
            fa_test[l] = f
            qf = compute_q(fa_test, kd)[0][l]
            print(f"    q(l={l}, f={f:.4f}) = {qf:.6f}")
        break
    J = [[0.0]*len(active) for _ in range(len(active))]
    for jj, l in enumerate(active):
        for kk, m in enumerate(active):
            t = 0.0
            for j in range(1, 7):
                if uaj[l][j] == 0: continue
                t += uaj[l][j] * caj[m][j] * rj[j]**2 * (kd[m][j] - 1.0)
            J[jj][kk] = -t
    rhs = [q[l] for l in active]
    dfa_active = solve_linear(J, rhs)
    if dfa_active is None:
        print("  singular Jacobian")
        break
    # Fortran uses (-J)*x = q, so x = -J^{-1}*q, update fa = fa + alpha*x = fa - alpha*J^{-1}*q
    print(f"  J^{-1}q={dfa_active}")
    for i, l in enumerate(active):
        fa[l] = max(0.0, min(0.9999, fa[l] - dfa_active[i]))
    if sum(fa[1:4]) > 0.9999:
        s = 0.9999 / sum(fa[1:4])
        for l in [1,2,3]: fa[l] *= s
    # update Kd based on new liquid composition (like Fortran kdcalc mode 3)
    kd = kdcalc(tk, clj)
