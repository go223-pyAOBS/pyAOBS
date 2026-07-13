import numpy as np

def delta_t_pps_minus_ppp_by_conversion_receiver_offset(
    H: float,      # 接收点到转换界面的垂向深度 (km)
    Vp: float,     # 转换段 P 波速度 (km/s)
    Vs: float,     # 转换段 S 波速度 (km/s)
    dx_cr: np.ndarray,  # 转换点到接收点的水平偏移距 (km)
) -> np.ndarray:
    """
    1D 近似：同一道 PPP/PPS 共享同一条射线，仅“转换点->接收点”最后一段波型不同。

    设转换点 C 到接收点 R 的几何长度:
        L_cr = sqrt(H^2 + dx_cr^2)
    则:
        t_PPP_last = L_cr / Vp
        t_PPS_last = L_cr / Vs
        Delta_t = t_PPS_last - t_PPP_last = L_cr * (1/Vs - 1/Vp)

    这个模型直接反映：随着 |dx_cr| 增大，L_cr 增大，PPS-PPP 单调增大。
    """
    dx_cr = np.asarray(dx_cr, dtype=float)
    if H <= 0 or Vp <= 0 or Vs <= 0:
        raise ValueError("H, Vp, Vs must be positive")
    if Vs >= Vp:
        raise ValueError("Vs should be smaller than Vp for typical sediments")

    l_cr = np.sqrt(H * H + dx_cr * dx_cr)
    dt = l_cr * (1.0 / Vs - 1.0 / Vp)
    return dt

# 简单测试：画出趋势
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    H, Vp, Vs = 2.0, 4.0, 2.3  # 示例参数
    dx = np.linspace(0, 5, 200)
    dt = delta_t_pps_minus_ppp_by_conversion_receiver_offset(H, Vp, Vs, dx)

    plt.plot(dx, dt, label="Δt = PPS-PPP (C-R segment)")
    plt.xlabel("Conversion-to-receiver offset |dx_cr| (km)")
    plt.ylabel("Δt (s)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()