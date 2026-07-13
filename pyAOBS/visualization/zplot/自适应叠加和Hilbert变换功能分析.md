# 自适应叠加和 Hilbert 变换功能分析

## 一、功能概述

Haibo Huang 在 2023年11月对 ZPLOT 代码进行了重要改进，主要涉及两个核心功能：

1. **自适应叠加对齐（tcas 子程序）**：改进的归一化处理和误差估计
2. **Hilbert 变换增强（hilbert 子程序）**：新增相位和包络提取功能
3. **叠加处理改进（pstack 子程序）**：结合 Hilbert 变换的相位一致性加权

这些改进显著提高了信号处理的稳定性和拾取质量评估的准确性。

---

## 二、自适应叠加对齐（tcas 子程序）

### 2.1 功能定位

**文件位置：** `zplot/misc.f`  
**子程序：** `subroutine tcas(nsi,pjgl,wb,dtcw,wl,dts,zv,ntpk,nptsk,ipltk,tshift,ratio)`

**功能描述：**
- 非线性反演方案，用于确定最优道对齐
- 给定初始近似对齐（如模型预测的 AK135），可获得模型走时残差
- 使用 Lp 范数进行自适应叠加

**历史版本：**
- 原始作者：Brian L.N. Kennett (ANU, 2002)
- 第一次修改：Nick Rawlinson (RSES, ANU, 2003年8月)
- **本次修改：** Haibo Huang (SCSIO, CAS, 2023年11月)

### 2.2 核心算法流程

#### 步骤1：数据归一化（改进点）

```fortran
DO i=1,nsta
    scu = 0.
    DO j=1,nptsk
        zu(i,j) = w(i)*zv(i,j)
        if(abs(zu(i,j)).gt.scu) scu = abs(zu(i,j))
    ENDDO
    
    ! 归一化处理（Haibo 改进）
    IF(scu.gt.0.0) THEN
        DO j=1,nptsk
            zu(i,j) = zu(i,j)/scu  ! 使用最大绝对值归一化
        ENDDO
    ENDIF
ENDDO
```

**改进点：**
- 对每条道进行独立归一化
- 使用最大绝对值进行归一化，避免小振幅道被大振幅道压制
- 提高了叠加的稳定性，特别是对于振幅差异较大的 OBS 数据

#### 步骤2：初始叠加

```fortran
call pstack(nstkwb,nstkwl,nsta,dts,ratio)
```

使用初步对齐的道进行初始叠加，生成参考叠加道 `zssl`。

#### 步骤3：迭代对齐优化

```fortran
DO m = 1,nsi  ! nsi = 叠加迭代次数（默认10次）
    DO i = 1,nsta
        ! 在搜索范围内寻找最优对齐
        DO js = jim1,jim2  ! jim1到jim2是搜索范围
            ws = 0.
            DO l = 1,nstkwl
                lu = l-nstkwb+imo+js-1
                ! 使用Lp范数计算差异
                ws = ws + abs(zssl(l)-zu(i,lu))**pjgl
            ENDDO
            ws = ws/stkwl
            
            ! 记录最小差异对应的偏移
            IF(ws.lt.wm) THEN
                wm = ws
                jm = js
            ENDIF
        ENDDO
        
        ! 确定时间偏移
        dtcs(i) = float(jm)*dts
        tshift(i) = dtcs(i)
    ENDDO
    
    ! 使用新的对齐重新叠加
    call pstack(nstkwb,nstkwl,nsta,dts,ratio)
ENDDO
```

**算法特点：**
- **Lp 范数**：`pjgl` 参数控制范数类型（默认3）
  - `pjgl=1`：L1范数（对异常值鲁棒）
  - `pjgl=2`：L2范数（最小二乘）
  - `pjgl=3`：L3范数（更强调大差异）
- **迭代优化**：每次迭代都更新叠加道，逐步收敛到最优对齐
- **搜索范围**：`dtcw` 参数控制搜索范围（差分时间搜索边界）

#### 步骤4：误差估计（Haibo 改进）

```fortran
if(m.eq.nsi)then  ! 最后一次迭代时计算误差
    ! 计算叠加功率的半宽度
    ! 找到功率达到最小值*erl倍的位置
    
    ! 左侧交叉点
    do while(swl.eq.0)
        if(wsp(l).ge.wsp(jmi)*erl)then
            ! 线性插值计算精确位置
            errl = ...
        endif
    enddo
    
    ! 右侧交叉点
    do while(swr.eq.0)
        if(wsp(l).ge.wsp(jmi)*erl)then
            errr = ...
        endif
    enddo
    
    ! 平均误差
    err(i) = (errr+errl)/2.0
    
    ! 约束误差范围
    if(err(i).lt.emin) err(i)=emin  ! emin=0.025
    if(err(i).gt.emax) err(i)=emax  ! emax=0.150
endif
```

**改进点：**
- **半宽度误差估计**：基于叠加功率曲线的半宽度计算误差
- **线性插值**：提高误差估计的精度
- **误差约束**：限制误差在合理范围内（0.025-0.150秒）

### 2.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nsi` | 10 | 叠加迭代次数 |
| `pjgl` | 3 | 叠加指数（Lp范数） |
| `stkwb` | 0.04 | 叠加窗口起始时间（秒） |
| `stkwl` | 窗口长度 | 叠加窗口长度（秒） |
| `dtcw` | 搜索范围 | 差分时间搜索边界 |
| `emin` | 0.025 | 最小误差（秒） |
| `emax` | 0.150 | 最大误差（秒） |
| `erl` | 1.25 | 误差估计比例因子 |

### 2.4 改进效果

1. **更稳定的道对齐**
   - 改进的归一化方法避免了振幅差异导致的偏差
   - 特别适合处理 OBS 数据（不同 OBS 的振幅可能差异很大）

2. **更准确的误差估计**
   - 基于叠加功率的半宽度方法更科学
   - 线性插值提高了精度
   - 误差约束保证了合理性

3. **更好的收敛性**
   - 迭代过程中逐步优化对齐
   - 每次迭代都更新参考叠加道

---

## 三、Hilbert 变换改进（hilbert 子程序）

### 3.1 功能定位

**文件位置：** `zplot/bandpass.f`  
**子程序：** `subroutine hilbert(x,n,ih)`

**功能描述：**
- 计算信号的 Hilbert 变换
- 提取信号的瞬时相位和包络
- 为互相关和信号分析提供支持

**历史版本：**
- 原始作者：Jonathas Maciel
- **本次修改：** Haibo Huang, 2023.11

### 3.2 算法实现

#### Hilbert 变换计算

```fortran
do it = 1,n
    soma = 0.0
    do i=-n,n
        if (it.eq.i) goto 42  ! 跳过奇异点
        soma = soma + buffer(i)/float(i-it)  ! 卷积积分
    enddo
    soma = soma/pi  ! 除以π得到Hilbert变换
    
    ! Haibo 新增：计算相位和包络
    theta = soma/x(it)
    evelp = sqrt(x(it)**2+soma**2)  ! 包络
    phase = atan(theta)              ! 相位
    
    ! 输出选项
    if(ih.eq.1) x(it) = phase   ! 输出相位
    if(ih.eq.2) x(it) = evelp    ! 输出包络
enddo
```

**数学原理：**
- Hilbert 变换：$H[x(t)] = \frac{1}{\pi} \int_{-\infty}^{\infty} \frac{x(\tau)}{t-\tau} d\tau$
- 解析信号：$z(t) = x(t) + iH[x(t)]$
- 包络：$A(t) = \sqrt{x(t)^2 + H[x(t)]^2}$
- 相位：$\phi(t) = \arctan\left(\frac{H[x(t)]}{x(t)}\right)$

### 3.3 改进点

1. **新增相位和包络计算**
   - 可以提取信号的瞬时相位
   - 可以提取信号的包络（振幅调制）

2. **增强的输出选项**
   - `ih=0`：标准 Hilbert 变换
   - `ih=1`：输出相位
   - `ih=2`：输出包络

3. **改进的缓冲区处理**
   - 优化了边界条件处理
   - 避免了数组越界问题

### 3.4 应用场景

1. **相位一致性分析**
   - 用于评估多道信号的相位一致性
   - 在叠加处理中作为权重因子

2. **包络提取**
   - 提取信号的振幅调制信息
   - 用于识别信号的到达时间

3. **互相关分析**
   - Hilbert 变换后的信号用于互相关
   - 提高互相关的稳定性

---

## 四、叠加处理改进（pstack 子程序）

### 4.1 功能定位

**文件位置：** `zplot/misc.f`  
**子程序：** `subroutine pstack(pnkwb,pnkwl,pnsta,pdts,ratio)`

**功能描述：**
- 对多道进行叠加处理
- 结合 Hilbert 变换的相位一致性加权
- 提高信噪比，增强弱信号

### 4.2 核心算法（Haibo 改进版）

#### 步骤1：线性叠加和二次叠加

```fortran
DO i = 1,pnsta
    IF(nst0(i).gt.0)then
        nst=nst+1
        lmn = -pnkwb + nst0(i) + nint(dtcs(i)/pdts) - 1
        
        DO l=1,pnkwl
            zcu(l) = zu(i,lmn+l)
            tmpp(l) = zu(i,lmn+l)
            
            ! 线性叠加
            zssl(l) = zssl(l) + zcu(l)
            ! 二次叠加（用于误差估计）
            zscp(l) = zscp(l) + zcu(l)*zcu(l)
        ENDDO
    ENDIF
ENDDO
```

#### 步骤2：Hilbert 变换相位一致性加权（Haibo 新增）

```fortran
! 对每道进行Hilbert变换，提取相位
call hilbert(tmpp,pnkwl,1)  ! ih=1，输出相位

! 计算相位一致性
DO l=1,pnkwl
    tempc(l) = tempc(l) + cos(tmpp(l))  ! 相位余弦和
    temps(l) = temps(l) + sin(tmpp(l))  ! 相位正弦和
ENDDO

! 计算相位一致性因子（归一化）
DO l=1,pnkwl
    tempf(l) = sqrt(tempc(l)**2+temps(l)**2)/nst  ! 相位一致性
    tempf(l) = tempf(l)**ratio  ! ratio次方（增强效果）
ENDDO
```

**数学原理：**
- 相位一致性：$C = \frac{1}{N}\left|\sum_{i=1}^{N} e^{i\phi_i}\right| = \frac{1}{N}\sqrt{\left(\sum\cos\phi_i\right)^2 + \left(\sum\sin\phi_i\right)^2}$
- 一致性越高，$C$ 越接近 1
- 一致性越低，$C$ 越接近 0

#### 步骤3：加权叠加

```fortran
! 使用相位一致性加权叠加
DO l=1,pnkwl
    zssl(l) = zssl(l)*tempf(l)/(real(nst))
ENDDO

! 归一化
scz = 0.
DO l=1,pnkwl
    IF(abs(zssl(l)).gt.scz) scz = abs(zssl(l))
ENDDO
IF(scz.gt.0.0) THEN
    DO l=1,pnkwl
        zssl(l) = zssl(l)/scz
    ENDDO
ENDIF
```

**改进效果：**
- **相位一致性加权**：相位一致的道权重更大，提高叠加质量
- **抑制噪声**：相位不一致的噪声被抑制
- **增强信号**：同相位的信号被增强

### 4.3 误差估计

```fortran
! L2测度的道失配
pstakn = 0.
DO l=1,pnkwl
    pstakn = pstakn + abs(zscp(l))
ENDDO
pstakn = pstakn/(real(nst)*pnkwl)
write(6,*) "pstakn = ", pstakn
```

**用途：**
- 评估叠加质量
- 用于误差估计和不确定性分析

---

## 五、三个功能的协同工作

### 5.1 工作流程

```
1. tcas 初始化
   ↓
2. 数据归一化（改进的归一化方法）
   ↓
3. 初始叠加（pstack）
   ├─ 线性叠加
   ├─ 二次叠加
   └─ Hilbert变换相位一致性加权（新增）
   ↓
4. 迭代对齐优化（nsi次迭代）
   ├─ 计算每道与叠加道的差异（Lp范数）
   ├─ 寻找最优对齐
   └─ 更新叠加道
   ↓
5. 误差估计（最后一次迭代）
   ├─ 基于叠加功率的半宽度
   └─ 线性插值提高精度
   ↓
6. 输出时间偏移和误差估计
```

### 5.2 协同效应

1. **归一化 + 相位一致性加权**
   - 归一化处理振幅差异
   - 相位一致性加权处理相位差异
   - 两者结合，显著提高叠加稳定性

2. **Hilbert变换 + 叠加**
   - Hilbert变换提取相位信息
   - 相位一致性作为权重因子
   - 提高叠加质量，抑制噪声

3. **迭代优化 + 误差估计**
   - 迭代过程逐步优化对齐
   - 误差估计评估质量
   - 形成完整的质量控制体系

---

## 六、改进的技术优势

### 6.1 处理稳定性提升

1. **归一化改进**
   - 避免大振幅道压制小振幅道
   - 特别适合 OBS 数据（振幅差异大）

2. **相位一致性加权**
   - 自动识别同相位信号
   - 抑制随机噪声和异常道

3. **迭代优化**
   - 逐步收敛到最优解
   - 避免局部最优

### 6.2 误差估计改进

1. **半宽度方法**
   - 基于叠加功率曲线的物理意义
   - 比简单的统计方法更准确

2. **线性插值**
   - 提高误差估计的精度
   - 避免离散采样带来的误差

3. **误差约束**
   - 保证误差在合理范围内
   - 避免异常值影响

### 6.3 OBS 数据处理优势

1. **多 OBS 支持**
   - 可以处理不同 OBS 的振幅差异
   - 可以处理不同 OBS 的相位差异

2. **弱信号增强**
   - 相位一致性加权增强弱信号
   - 提高信噪比

3. **质量评估**
   - 误差估计帮助评估拾取质量
   - 为后续处理提供参考

---

## 七、代码实现细节

### 7.1 关键数据结构

```fortran
common /RST1/ zu(1000,npmax),zssl(npmax),zscp(npmax)
common /RST2/ dtcs(1000),nst0(1000)

! zu: 归一化后的道数据
! zssl: 线性叠加结果
! zscp: 二次叠加结果（用于误差估计）
! dtcs: 每道的时间偏移
! nst0: 每道的初始拾取样本号
```

### 7.2 关键变量

| 变量 | 说明 |
|------|------|
| `tempc(l)` | 相位余弦和（用于相位一致性计算） |
| `temps(l)` | 相位正弦和（用于相位一致性计算） |
| `tempf(l)` | 相位一致性因子（加权系数） |
| `pstakn` | L2测度的道失配（误差指标） |
| `wsp(js)` | 叠加功率（用于误差估计） |
| `err(i)` | 每道的误差估计 |

### 7.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ratio` | 叠加比例因子 | 控制相位一致性加权的强度 |
| `erl` | 1.25 | 误差估计比例因子（半宽度阈值） |
| `emin` | 0.025 | 最小误差（秒） |
| `emax` | 0.150 | 最大误差（秒） |

---

## 八、使用示例

### 8.1 调用方式

```fortran
! 在主程序中调用
call tcas(nsi,pjgl,stkwb,tlag,tcrcor,sints,zv,nt0,
     +     nplt,iplt,tshift,hilbratio)

! 参数说明：
! nsi=10      : 叠加迭代次数
! pjgl=3      : Lp范数指数
! stkwb=0.04  : 叠加窗口起始时间（秒）
! tlag        : 时间延迟搜索范围
! tcrcor      : 互相关窗口长度
! sints       : 采样间隔
! zv          : 输入道数据
! nt0         : 初始拾取样本号
! nplt        : 每道数据点数
! iplt        : 道数
! tshift      : 输出时间偏移
! hilbratio   : Hilbert变换比例因子
```

### 8.2 输出结果

1. **tshift(i)**：每道的时间偏移（秒）
2. **err(i)**：每道的误差估计（秒）
3. **zssl(l)**：最终叠加结果
4. **pstakn**：叠加质量指标

---

## 九、改进效果总结

### 9.1 稳定性提升

1. **归一化改进**
   - ✅ 避免振幅差异导致的偏差
   - ✅ 提高叠加稳定性
   - ✅ 特别适合 OBS 数据

2. **相位一致性加权**
   - ✅ 自动识别同相位信号
   - ✅ 抑制噪声和异常道
   - ✅ 提高信噪比

### 9.2 误差估计改进

1. **半宽度方法**
   - ✅ 基于物理意义的方法
   - ✅ 比统计方法更准确
   - ✅ 线性插值提高精度

2. **误差约束**
   - ✅ 保证误差在合理范围
   - ✅ 避免异常值

### 9.3 功能扩展

1. **Hilbert变换增强**
   - ✅ 相位提取
   - ✅ 包络提取
   - ✅ 为信号分析提供支持

2. **质量评估**
   - ✅ 误差估计
   - ✅ 叠加质量指标
   - ✅ 为后续处理提供参考

---

## 十、算法流程图

### 10.1 自适应叠加对齐流程

```
开始
 ↓
初始化参数（nsi, pjgl, wb, wl, dtcw）
 ↓
数据归一化（改进方法）
 ├─ 对每条道独立归一化
 └─ 使用最大绝对值归一化
 ↓
初始叠加（pstack）
 ├─ 线性叠加：zssl = Σ zcu
 ├─ 二次叠加：zscp = Σ zcu²
 └─ Hilbert变换相位一致性加权
    ├─ 提取每道相位
    ├─ 计算相位一致性：C = |Σe^(iφ)|/N
    └─ 加权叠加：zssl = zssl * C^ratio
 ↓
迭代优化（m = 1 to nsi）
 ├─ 对每道 i：
 │   ├─ 在搜索范围内（jim1 到 jim2）
 │   │   ├─ 计算差异：ws = Σ|zssl - zu|^pjgl
 │   │   └─ 找到最小差异对应的偏移 jm
 │   ├─ 更新时间偏移：dtcs(i) = jm * dts
 │   └─ （最后一次迭代）计算误差估计
 │       ├─ 计算叠加功率半宽度
 │       ├─ 线性插值提高精度
 │       └─ 约束误差范围（emin 到 emax）
 └─ 使用新对齐重新叠加
 ↓
输出结果
 ├─ tshift(i)：时间偏移
 └─ err(i)：误差估计
```

### 10.2 Hilbert变换相位一致性加权流程

```
输入：多道信号 zu(i,l)
 ↓
对每道进行Hilbert变换
 ├─ 计算Hilbert变换：H[x] = (1/π)∫x(τ)/(t-τ)dτ
 ├─ 提取相位：φ = arctan(H[x]/x)
 └─ 提取包络：A = √(x² + H[x]²)
 ↓
计算相位一致性
 ├─ 相位余弦和：tempc = Σcos(φ)
 ├─ 相位正弦和：temps = Σsin(φ)
 └─ 一致性因子：C = √(tempc² + temps²)/N
 ↓
加权叠加
 ├─ 应用权重：zssl = zssl * C^ratio
 └─ 归一化输出
```

### 10.3 误差估计流程

```
最后一次迭代
 ↓
计算叠加功率曲线 wsp(js)
 ├─ 对每个搜索位置 js：
 │   └─ wsp(js) = Σ|zssl - zu(js)|^pjgl
 ↓
找到最小值位置 jmi
 ↓
计算半宽度
 ├─ 左侧交叉点（errl）
 │   ├─ 找到 wsp(l) ≥ wsp(jmi) * erl 的位置
 │   └─ 线性插值计算精确位置
 └─ 右侧交叉点（errr）
     ├─ 找到 wsp(l) ≥ wsp(jmi) * erl 的位置
     └─ 线性插值计算精确位置
 ↓
计算平均误差
 └─ err = (errl + errr) / 2
 ↓
约束误差范围
 ├─ if err < emin: err = emin
 └─ if err > emax: err = emax
```

---

## 十一、数学公式详解

### 11.1 Lp 范数

$$L_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

在代码中使用：
$$ws = \frac{1}{wl}\sum_{l=1}^{wl} |zssl(l) - zu(i,lu)|^{pjgl}$$

- `pjgl=1`：L1范数（对异常值鲁棒）
- `pjgl=2`：L2范数（最小二乘，标准）
- `pjgl=3`：L3范数（更强调大差异，Haibo默认）

### 11.2 Hilbert 变换

离散形式：
$$H[x(n)] = \frac{1}{\pi}\sum_{m=-N}^{N} \frac{x(m)}{n-m}, \quad m \neq n$$

解析信号：
$$z(n) = x(n) + iH[x(n)]$$

包络：
$$A(n) = |z(n)| = \sqrt{x(n)^2 + H[x(n)]^2}$$

相位：
$$\phi(n) = \arg[z(n)] = \arctan\left(\frac{H[x(n)]}{x(n)}\right)$$

### 11.3 相位一致性

对于 N 道信号，相位一致性定义为：
$$C = \frac{1}{N}\left|\sum_{i=1}^{N} e^{i\phi_i}\right| = \frac{1}{N}\sqrt{\left(\sum_{i=1}^{N}\cos\phi_i\right)^2 + \left(\sum_{i=1}^{N}\sin\phi_i\right)^2}$$

- $C \in [0, 1]$
- $C = 1$：完全同相
- $C = 0$：完全随机相位

加权叠加：
$$z_{stack}(l) = \frac{1}{N}\sum_{i=1}^{N} z_i(l) \cdot C(l)^{ratio}$$

### 11.4 误差估计（半宽度方法）

基于叠加功率曲线的半宽度：
- 找到功率最小值位置：$j_{mi}$
- 找到功率达到最小值 $\times erl$ 的位置：$j_l$ 和 $j_r$
- 线性插值计算精确位置：
  $$err_l = (j_{mi} - j_l) \cdot dt - \frac{wsp(j_{mi}) \cdot erl - wsp(j_l)}{wsp(j_l+1) - wsp(j_l)} \cdot dt$$
- 平均误差：
  $$err = \frac{err_l + err_r}{2}$$

---

## 十二、改进前后对比

### 12.1 归一化方法对比

**改进前（原始版本）：**
- 可能使用全局归一化或简单平均
- 大振幅道可能压制小振幅道

**改进后（Haibo版本）：**
- 每条道独立归一化
- 使用最大绝对值归一化
- 避免振幅差异导致的偏差

### 12.2 叠加方法对比

**改进前：**
- 简单线性叠加：$z_{stack} = \frac{1}{N}\sum z_i$

**改进后：**
- 相位一致性加权叠加：$z_{stack} = \frac{1}{N}\sum z_i \cdot C_i^{ratio}$
- 自动识别同相位信号
- 抑制噪声和异常道

### 12.3 误差估计对比

**改进前：**
- 可能使用简单的统计方法
- 误差估计不够准确

**改进后：**
- 基于叠加功率曲线的半宽度方法
- 线性插值提高精度
- 误差约束保证合理性

---

## 十三、参考文献

- **tcas 子程序原始作者**：Brian L.N. Kennett (ANU, 2002)
- **tcas 子程序第一次修改**：Nick Rawlinson (RSES, ANU, 2003)
- **本次修改**：Haibo Huang (SCSIO, CAS, 2023年11月)
- **Hilbert 变换原始作者**：Jonathas Maciel
- **Hilbert 变换修改**：Haibo Huang (2023年11月)

---

**文档生成日期：** 2024年  
**基于代码版本：** ZPLOT v3.0 (Haibo Huang 2023修改版)
