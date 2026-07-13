import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import StringIO
# ===================== 1. 固定窗口参数 =====================
alpha_fix = 0.6
beta_fix = 8.4
Pt_fix = 1.0
Ft_fix = 0.05

def W_L(P, F):
    tp = 1 - np.tanh(alpha_fix * (P - Pt_fix))
    tf = 1 - np.tanh(beta_fix * (F - Ft_fix))
    return 0.25 * tp * tf

def W_H(P, F):
    tp = 1 + np.tanh(alpha_fix * (P - Pt_fix))
    tf = 1 + np.tanh(beta_fix * (F - Ft_fix))
    return 0.25 * tp * tf

# 修复model：不构造np.array，直接手动多项式求和避免维度冲突
def model(x, P, F):
    a0 = x[0]
    b0,b1,b2,b3,b4,b5 = x[1:7]
    c0,c1,c2,c3,c4,c5 = x[7:13]
    # 多项式项
    p1 = P
    f1 = F
    p2 = P**2
    pf = P*F
    f2 = F**2
    poly_b = b0 + b1*p1 + b2*f1 + b3*p2 + b4*pf + b5*f2
    poly_c = c0 + c1*p1 + c2*f1 + c3*p2 + c4*pf + c5*f2
    WL = W_L(P, F)
    WH = W_H(P, F)
    return a0 + WL * poly_b + WH * poly_c

# 损失函数
def loss_func(params, P_arr, F_arr, V_true_arr):
    total_loss = 0.0
    for p, f, vt in zip(P_arr, F_arr, V_true_arr):
        vm = model(params, p, f)
        total_loss += (vm - vt) ** 2
    return total_loss

# ===================== 2. 数字化等值线数据 =====================
data_text = """
6.9
 5.28741267863489e-001    5.22349321446082e-002
 6.25081529928794e-001    4.97060424360468e-002
 7.35134563741785e-001    4.46320694882394e-002
 9.00272591099856e-001    3.95688922275551e-002
 1.05161013175237e+000    3.32291249645766e-002
 1.23051940749671e+000    2.81686466256730e-002
 1.47822644853381e+000    2.05738807346465e-002
 1.62956398918632e+000    1.42341134716681e-002
7.0
 5.15934884013836e-001    9.42706388797675e-002
 6.12128954482684e-001    8.53722937686394e-002
 7.49607531791050e-001    7.52081543423402e-002
 9.00828119166393e-001    6.37728227573085e-002
 1.06587843156659e+000    5.48879722550841e-002
 1.29975574757884e+000    4.47427252812501e-002
 1.56114632204434e+000    3.33289850704643e-002
 1.79499439973731e+000    2.19098470161171e-002
 1.97384519884306e+000    1.43015865161868e-002
7.1
 5.44793105154491e-001    1.51601135346429e-001
 6.95867500933377e-001    1.33796348358831e-001
 8.60713145098533e-001    1.15994260293013e-001
 9.98104007449024e-001    1.02008447625174e-001
 1.16297888993347e+000    8.54802506398694e-002
 1.38296800428229e+000    7.02367404222013e-002
 1.60301559526969e+000    5.75410123655600e-002
 1.80929193787082e+000    4.48425853871378e-002
 2.02933952885822e+000    3.21468573304965e-002
 2.34566890527234e+000    1.43744574042670e-002
7.2
 6.83587406830971e-001    1.98762094543230e-001
 8.62321252659563e-001    1.86058269721246e-001
 1.02725461178259e+000    1.72077854896969e-001
 1.28835280305518e+000    1.47925203881050e-001
 1.53547507770646e+000    1.14852616379756e-001
 1.75525952382023e+000    9.06918685984949e-002
 2.05767146025163e+000    6.65473143479180e-002
 2.36014187332161e+000    4.49505422583678e-002
 2.66264152471088e+000    2.46276612493309e-002
 2.80014934033853e+000    1.57374129035450e-002
7.3
 8.22586376742491e-001    2.54840291303624e-001
 1.04269244436847e+000    2.44692345408009e-001
 1.33153780064864e+000    2.29462329799245e-001
 1.64777946210489e+000    2.07868256631476e-001
 1.90884841505818e+000    1.82441714535043e-001
 2.16977117641502e+000    1.50645717036044e-001
 2.38940943093234e+000    1.20115513852216e-001
 2.59542262865985e+000    9.59520671491739e-002
 2.84266185658829e+000    6.79750439699340e-002
 3.04879200759297e+000    4.89071615889452e-002
 3.33737421899951e+000    2.22121262555609e-002
7.4
 1.19651524216076e+000    3.46633319988664e-001
 1.56822199531287e+000    3.41610626554691e-001
 1.87101402989506e+000    3.34026656350788e-001
 2.17377682615795e+000    3.25168795066371e-001
 2.44893864900972e+000    3.13757753777366e-001
 2.72398351858433e+000    2.97251148166308e-001
 2.94394339461385e+000    2.80733746868126e-001
 3.24647228432241e+000    2.61684756939603e-001
 3.47996950218388e+000    2.34978925919095e-001
 3.87819541093333e+000    1.85375442510717e-001
 4.06991107052732e+000    1.38279257436654e-001
 4.16516951477885e+000    8.86163977490992e-002
 4.20584001691324e+000    6.05988907431481e-002
 4.28791197916432e+000    3.64111537440792e-002
7.5
 2.59168012379055e+000    5.32894008843467e-001
 2.94935248368277e+000    5.16403596763093e-001
 3.29325359518872e+000    4.99910485760938e-001
 3.54075596799079e+000    4.83398482306319e-001
 3.93959588144536e+000    4.60546711588720e-001
 4.29694661982538e+000    4.30043497622700e-001
 4.73680789524585e+000    3.94460912865310e-001
 5.23143254269919e+000    3.44876321909397e-001
 5.58846165956700e+000    3.00360306057730e-001
 5.90408931631813e+000    2.52014520199180e-001
 6.21971697306926e+000    2.03668734340631e-001
 6.48023039795602e+000    1.54038261714445e-001
 6.64460822901251e+000    1.15853916360414e-001
7.6
 5.69173140330438e+000    5.99743602430829e-001
 5.87040677249439e+000    5.84491995447819e-001
 6.11770447706141e+000    5.59062754429605e-001
 6.37874419169542e+000    5.32362321252659e-001
 6.65343820143852e+000    5.00569022675441e-001
 6.84568015077976e+000    4.76402877050618e-001
 6.99666683160078e+000    4.54776416821480e-001
"""

stream = StringIO(data_text)
digit_contours = {}
current_v = None
P_train = []
F_train = []
V_train = []
for line in stream:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) == 1:
        current_v = float(parts[0])
        digit_contours[current_v] = [[], []]
    else:
        p = float(parts[0])
        f = float(parts[1])
        digit_contours[current_v][0].append(p)
        digit_contours[current_v][1].append(f)
        P_train.append(p)
        F_train.append(f)
        V_train.append(current_v)

P_train = np.array(P_train)
F_train = np.array(F_train)
V_train = np.array(V_train)
print(f"总训练采样点数: {len(P_train)}")

# ===================== 3. 拟合优化 =====================
x0 = [7.52,
      -1.73, 0.55, 7.71, -0.11, 8.87, -146.11,
      -0.35, 0.034, 0.51, 0.0016, -0.040, 0.0]

res = minimize(loss_func, x0, args=(P_train, F_train, V_train), method="L-BFGS-B")
print("\n========== 拟合结果输出 ==========")
print("优化是否收敛:", res.success)
print("最小总残差平方和:", round(res.fun, 6))
opt_params = res.x
a0_opt = opt_params[0]
b_opt = opt_params[1:7]
c_opt = opt_params[7:13]
print(f"拟合得到 a0 = {round(a0_opt, 4)}")
print(f"拟合 b 系数: {np.round(b_opt, 4)}")
print(f"拟合 c 系数: {np.round(c_opt, 4)}")

# 封装标量专用速度函数
def V_fit_scalar(P, F):
    return model(opt_params, P, F)

# ===================== 4. 网格绘图（双重循环只传标量，彻底规避数组报错） =====================
grid_res = 350
P_arr = np.linspace(0.5, 7.0, grid_res)
F_arr = np.linspace(0.0, 0.6, grid_res)
P_mesh, F_mesh = np.meshgrid(P_arr, F_arr)
V_field = np.zeros_like(P_mesh)

# 逐点标量计算，不再传入数组进model
for i in range(grid_res):
    for j in range(grid_res):
        p_val = float(P_mesh[i,j])
        f_val = float(F_mesh[i,j])
        V_field[i,j] = V_fit_scalar(p_val, f_val)

# 绘图
plt.figure(figsize=(11,7))
pcm = plt.pcolormesh(P_mesh, F_mesh, V_field, cmap="gray", vmin=6.8, vmax=7.6, shading="auto")
plt.gca().invert_yaxis()
cs = plt.contour(P_mesh, F_mesh, V_field, levels=np.arange(6.8,7.61,0.1), colors="black", linewidth=0.8)
plt.clabel(cs, inline=True, fontsize=8)
first_line = True
for vel, (px, fx) in digit_contours.items():
    plt.plot(px, fx, "r-", lw=1.2, label="Paper digitized contour" if first_line else "")
    plt.text(px[-1]+0.07, fx[-1], f"{vel}", c="red", fontsize=8)
    first_line = False
plt.xlabel(r"$\overline{P}$ (GPa)")
plt.ylabel(r"$\overline{F}$")
plt.title("Fitted model(black) vs Original paper contour(red)")
plt.legend(loc="upper right")
plt.colorbar(pcm, label=r"$V_{bulk}\ \mathrm{km/s}$")
plt.tight_layout()
plt.show()

# ===================== 5. 抽样校验 =====================
print("\n========== 随机样本误差校验 ==========")
idx_sample = np.random.choice(len(P_train), 8, replace=False)
for i in idx_sample:
    p, f, v_true = P_train[i], F_train[i], V_train[i]
    v_pred = V_fit_scalar(p, f)
    print(f"P={p:.3f}, F={f} | 真值={v_true:.2f}, 拟合={v_pred:.3f}, 误差={v_pred-v_true:.3f}")