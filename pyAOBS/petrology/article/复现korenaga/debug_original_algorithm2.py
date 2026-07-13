import math

from kd_legacy import kdcalc_fortran


def run_case(cs, ti, tf, dt):
    ta = [0.0, 1.0, 0.666667, 1.0]
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

    def kdcalc(tk, kdmode, clj=None):
        # Original used BASALT.FOR: 10**(A/T) + B
        return kdcalc_fortran(tk, kdmode=kdmode, clj_1based=clj, form="basalt1990")

    def solve_linear(A, b):
        n = len(A)
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

    fa = [0.0]*4
    temp = ti
    results = []
    while True:
        tk = temp
        kd = kdcalc(tk, 3, [cs[j] for j in range(8)])
        max_iter = 20
        tol = 1e-5
        converged = False
        for it in range(max_iter):
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
            qa = [0.0]*4
            active = []
            for l in range(1, 4):
                qa[l] = -ta[l]
                for j in range(1, 7):
                    qa[l] += uaj[l][j] * caj[l][j]
                if qa[l] <= 0.0:
                    if l == 3:
                        t = 2*(caj[3][3]+caj[3][4]) + caj[3][5]
                        if t >= 1.0:
                            active.append(l)
                    else:
                        active.append(l)
            if len(active) == 0:
                converged = True
                break
            J = [[0.0]*len(active) for _ in range(len(active))]
            for jj, l in enumerate(active):
                for kk, m in enumerate(active):
                    t = 0.0
                    for j in range(1, 7):
                        if uaj[l][j] == 0: continue
                        t += uaj[l][j] * caj[m][j] * rj[j]**2 * (kd[m][j] - 1.0)
                    J[jj][kk] = -t
            rhs = [qa[l] for l in active]
            dfa = solve_linear(J, rhs)
            if dfa is None:
                print(f"T={tk:.2f} singular")
                break
            tst = 0.0
            for i, l in enumerate(active):
                fa[l] = max(0.0, min(0.9999, fa[l] + dfa[i]))
                tst += abs(dfa[i])
            if tst <= tol:
                converged = True
                break
            kd = kdcalc(tk, 3, clj)
        else:
            print(f"T={tk-273.16:.1f}C MAX ITER")
            break
        fl = 1.0 - sum(fa[1:4])
        results.append((tk-273.16, fl, fa[1], fa[2], fa[3]))
        if not converged:
            break
        temp += dt
        if (tf - temp) * dt <= 0:
            break
    return results

print("=== data_original (low MgO/FeO) ===")
cs1 = [0.0, 0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28]
res = run_case(cs1, 1673.16, 1273.16, -10)
for r in res[:5] + res[-5:]:
    print(f"T={r[0]:.1f}C fl={r[1]:.4f} plag={r[2]:.4f} ol={r[3]:.4f} cpx={r[4]:.4f}")

print("\n=== data_highMgFe (high MgO/FeO) ===")
cs2 = [0.0, 0.15, 0.15, 0.20, 0.15, 0.10, 0.02, 0.23]
res = run_case(cs2, 1873.16, 1473.16, -10)
for r in res[:10]:
    print(f"T={r[0]:.1f}C fl={r[1]:.4f} plag={r[2]:.4f} ol={r[3]:.4f} cpx={r[4]:.4f}")
