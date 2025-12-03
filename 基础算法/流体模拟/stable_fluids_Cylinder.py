"""
对stable_fluids_python_simple 进行修改
增加圆柱体，并修改为非等边网格

1：将网格改成长方形，因此采样点个数需要根据物理比例自动调整
2：增加圆柱体，用于生成卡门涡街
"""

import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt

import cmasher as cmr
from tqdm import tqdm

# %matplotlib inline
from IPython.display import display, clear_output


# ----------------------------------------
# 物理域参数（修改为你需要的长方形域）
# ----------------------------------------
DOMAIN_SIZE_X = 2.0
DOMAIN_SIZE_Y = 0.5

# 采样点按照物理比例自动调整（保持与之前相同的密度）
# BASE_POINTS = 41
BASE_POINTS = 81
N_POINTS_X = int(BASE_POINTS * (DOMAIN_SIZE_X / 1.0))  # 大约 82 → 我改为奇数 81
N_POINTS_Y = int(BASE_POINTS * (DOMAIN_SIZE_Y / 1.0))  # 大约 20.5 → 21

# 调整为奇数点（保持中心点）
if N_POINTS_X % 2 == 0: N_POINTS_X += 1
if N_POINTS_Y % 2 == 0: N_POINTS_Y += 1

N_TIME_STEPS = 1000
TIME_STEP_LENGTH = 0.1
KINEMATIC_VISCOSITY = 0.0001
MAX_ITER_CG = None

# ----------------------------------------
# 圆柱体参数（用于卡门涡街模拟）
# ----------------------------------------
# 卡门涡街的形成依赖于雷诺数 Re = U*D/ν
# 其中 U 是来流速度，D 是圆柱体直径，ν 是运动粘度
# 典型的卡门涡街出现在 Re ≈ 40-200 之间
CYLINDER_CENTER_X = 0.5  # 圆柱体中心X坐标
CYLINDER_CENTER_Y = DOMAIN_SIZE_Y / 2.0  # 圆柱体中心Y坐标（居中）
CYLINDER_RADIUS = 0.05  # 圆柱体半径（直径 D = 0.1）
INFLOW_VELOCITY = 0.5  # 来流速度（从左到右）
# 当前雷诺数 Re ≈ INFLOW_VELOCITY * (2*CYLINDER_RADIUS) / KINEMATIC_VISCOSITY
# ≈ 1.0 * 0.1 / 0.0001 = 1000（较高，会产生湍流）
# 如需观察经典卡门涡街，可尝试：INFLOW_VELOCITY = 0.04, KINEMATIC_VISCOSITY = 0.001


# ----------------------------------------
# 网格构建
# ----------------------------------------
x = np.linspace(0.0, DOMAIN_SIZE_X, N_POINTS_X)
y = np.linspace(0.0, DOMAIN_SIZE_Y, N_POINTS_Y)

X, Y = np.meshgrid(x, y, indexing='ij')
coordinates = np.stack([X, Y], axis=-1)

element_length_x = DOMAIN_SIZE_X / (N_POINTS_X - 1)
element_length_y = DOMAIN_SIZE_Y / (N_POINTS_Y - 1)

scalar_shape = (N_POINTS_X, N_POINTS_Y)
vector_shape = (N_POINTS_X, N_POINTS_Y, 2)
scalar_dof = N_POINTS_X * N_POINTS_Y
vector_dof = N_POINTS_X * N_POINTS_Y * 2


# ----------------------------------------
# 微分算子（根据非等边网格修改）
# ----------------------------------------
def laplace(field):
    diff = np.zeros_like(field)

    dx2 = element_length_x ** 2
    dy2 = element_length_y ** 2

    diff[1:-1, 1:-1] = (
        (field[:-2, 1:-1] - 2*field[1:-1, 1:-1] + field[2:, 1:-1]) / dx2 +
        (field[1:-1, :-2] - 2*field[1:-1, 1:-1] + field[1:-1, 2:]) / dy2
    )
    return diff


def partial_derivative_x(field):
    diff = np.zeros_like(field)
    diff[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * element_length_x)
    return diff


def partial_derivative_y(field):
    diff = np.zeros_like(field)
    diff[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * element_length_y)
    return diff


def divergence(vector_field):
    return partial_derivative_x(vector_field[..., 0]) + partial_derivative_y(vector_field[..., 1])


def gradient(field):
    return np.stack([partial_derivative_x(field), partial_derivative_y(field)], axis=-1)


def curl_2d(vector_field):
    return partial_derivative_x(vector_field[..., 1]) - partial_derivative_y(vector_field[..., 0])


# ----------------------------------------
# 圆柱体检测
# ----------------------------------------
# 创建圆柱体掩码（标记圆柱体内部的网格点）
# 使用向量化计算，更高效
dx = X - CYLINDER_CENTER_X
dy = Y - CYLINDER_CENTER_Y
distance_squared = dx**2 + dy**2
cylinder_mask = distance_squared <= CYLINDER_RADIUS**2


# ----------------------------------------
# 半拉格朗日平流（保持原有）
# ----------------------------------------
def advect(field, vector_field):
    backtraced_positions = np.empty_like(vector_field)

    backtraced_positions[..., 0] = np.clip(
        coordinates[..., 0] - TIME_STEP_LENGTH * vector_field[..., 0],
        0.0, DOMAIN_SIZE_X
    )
    backtraced_positions[..., 1] = np.clip(
        coordinates[..., 1] - TIME_STEP_LENGTH * vector_field[..., 1],
        0.0, DOMAIN_SIZE_Y
    )

    result = interpolate.interpn(
        points=(x, y),
        values=field,
        xi=backtraced_positions,
        bounds_error=False,
        fill_value=0.0,  # 边界外填充为0
    )
    
    # 如果结果是标量场，确保圆柱体内部为0
    if result.ndim == 2:
        result[cylinder_mask] = 0.0
    # 如果是向量场，确保圆柱体内部速度分量为0
    elif result.ndim == 3 and result.shape[2] == 2:
        result[cylinder_mask, :] = 0.0
    
    return result


# ----------------------------------------
# 扩散、泊松算子
# ----------------------------------------
def diffusion_operator(v_flat):
    v = v_flat.reshape(vector_shape)
    return (v - KINEMATIC_VISCOSITY * TIME_STEP_LENGTH * laplace(v)).flatten()


def poisson_operator(p_flat):
    p = p_flat.reshape(scalar_shape)
    return laplace(p).flatten()



def forcing_function(time, point):
    # 对于卡门涡街模拟，不使用外力，只依赖边界条件
    # 如果需要外力，可以在这里添加
    return np.array([0.0, 0.0])



forcing_function_vectorized = np.vectorize(forcing_function, signature="(),(d)->(d)")


# ----------------------------------------
# 应用圆柱体边界条件
# ----------------------------------------
def apply_cylinder_boundary_conditions(velocity_field):
    """在圆柱体内部和边界应用无滑移边界条件（速度为零）"""
    velocity_field = velocity_field.copy()  # 避免修改原数组
    velocity_field[cylinder_mask, 0] = 0.0  # x方向速度为零
    velocity_field[cylinder_mask, 1] = 0.0  # y方向速度为零
    return velocity_field


def check_divergence(velocity_field, name="", threshold=1e-5, verbose=False):
    """检查速度场的散度，用于调试"""
    div = divergence(velocity_field)
    # 排除边界和圆柱体内部
    interior_mask = np.ones_like(div, dtype=bool)
    interior_mask[0, :] = False  # 左边界
    interior_mask[-1, :] = False  # 右边界
    interior_mask[:, 0] = False  # 下边界
    interior_mask[:, -1] = False  # 上边界
    interior_mask[cylinder_mask] = False  # 圆柱体内部
    
    if np.any(interior_mask):
        max_div = np.max(np.abs(div[interior_mask]))
        mean_div = np.mean(np.abs(div[interior_mask]))
        rms_div = np.sqrt(np.mean(div[interior_mask]**2))
    else:
        max_div = mean_div = rms_div = 0.0
    
    if verbose or max_div > threshold:
        print(f"{name}: 最大散度 = {max_div:.2e}, 平均散度 = {mean_div:.2e}, RMS散度 = {rms_div:.2e}")
    
    return max_div, mean_div, rms_div


# ----------------------------------------
# 主循环
# ----------------------------------------
plt.style.use('dark_background')
# 设置figure大小与物理域比例匹配（DOMAIN_SIZE_X : DOMAIN_SIZE_Y = 2.0 : 0.5 = 4:1）
plt.figure(figsize=(12, 3), dpi=160)

# 初始化速度场：添加均匀来流（从左到右）
velocities_prev = np.zeros(vector_shape)
velocities_prev[..., 0] = INFLOW_VELOCITY  # x方向初始速度

# 在圆柱体内部应用边界条件
velocities_prev = apply_cylinder_boundary_conditions(velocities_prev)

time_current = 0.0

for t in tqdm(range(N_TIME_STEPS)):
    time_current += TIME_STEP_LENGTH

    forces = forcing_function_vectorized(time_current, coordinates)
    v_forced = velocities_prev + TIME_STEP_LENGTH * forces
    
    # 应用圆柱体边界条件（在平流之前）
    v_forced = apply_cylinder_boundary_conditions(v_forced)

    v_adv = advect(v_forced, v_forced)
    
    # 应用圆柱体边界条件（在扩散之前）
    v_adv = apply_cylinder_boundary_conditions(v_adv)

    v_diff = splinalg.cg(
        A=splinalg.LinearOperator(shape=(vector_dof, vector_dof), matvec=diffusion_operator),
        b=v_adv.flatten(),
        maxiter=MAX_ITER_CG,
    )[0].reshape(vector_shape)
    
    # 应用圆柱体边界条件（在压力投影之前，这是关键！）
    v_diff = apply_cylinder_boundary_conditions(v_diff)
    
    # 计算散度用于压力求解
    div = divergence(v_diff)
    
    # 在圆柱体内部和边界设置散度为0（这些区域不参与压力求解）
    div[cylinder_mask] = 0.0  # 圆柱体内部散度设为0
    
    # 边界处的散度也设为0（边界条件由速度边界条件保证）
    div[0, :] = 0.0  # 左边界（入口，速度已固定）
    div[-1, :] = 0.0  # 右边界（出口，零梯度）
    div[:, 0] = 0.0  # 下边界（自由滑移）
    div[:, -1] = 0.0  # 上边界（自由滑移）

    pressure = splinalg.cg(
        A=splinalg.LinearOperator(shape=(scalar_dof, scalar_dof), matvec=poisson_operator),
        b=div.flatten(),
        maxiter=MAX_ITER_CG,
    )[0].reshape(scalar_shape)

    velocities_projected = v_diff - gradient(pressure)
    
    # 应用圆柱体边界条件（在投影之后，确保边界条件）
    velocities_projected = apply_cylinder_boundary_conditions(velocities_projected)
    
    # 边界条件设置
    # 左边界（入口）：固定来流速度
    velocities_projected[0, :, 0] = INFLOW_VELOCITY
    velocities_projected[0, :, 1] = 0.0
    
    # 右边界（出口）：零梯度边界条件（保持内部值）
    velocities_projected[-1, :, 0] = velocities_projected[-2, :, 0]
    velocities_projected[-1, :, 1] = velocities_projected[-2, :, 1]
    
    # 上下边界：自由滑移条件（法向速度为零，切向速度自由）
    velocities_projected[:, 0, 1] = 0.0  # 下边界，y方向速度为零
    velocities_projected[:, -1, 1] = 0.0  # 上边界，y方向速度为零
    
    # 检查散度（每10步打印一次，或当散度超过阈值时）
    if t % 10 == 0 or t < 5:
        max_div, mean_div, rms_div = check_divergence(
            velocities_projected, 
            f"t={time_current:.2f}", 
            threshold=1e-5,
            verbose=(t % 50 == 0)  # 每50步详细打印
        )
    else:
        max_div, mean_div, rms_div = check_divergence(
            velocities_projected, 
            f"t={time_current:.2f}", 
            threshold=1e-5,
            verbose=False
        )
    
    velocities_prev = velocities_projected

    curl = curl_2d(velocities_projected)
    
    # 在圆柱体内部将涡度设为NaN，以便在可视化中隐藏
    curl_vis = curl.copy()
    curl_vis[cylinder_mask] = np.nan

    plt.contourf(X, Y, curl_vis, cmap=cmr.redshift, levels=100)
    
    # 绘制圆柱体
    circle = plt.Circle((CYLINDER_CENTER_X, CYLINDER_CENTER_Y), 
                       CYLINDER_RADIUS, color='white', fill=True, zorder=10)
    plt.gca().add_patch(circle)
    
    # 绘制速度场（在圆柱体外部，使用下采样以提高性能）
    step = max(1, N_POINTS_X // 60)  # 下采样因子
    X_vis = X[::step, ::step]
    Y_vis = Y[::step, ::step]
    mask_vis = ~cylinder_mask[::step, ::step]
    u_vis = velocities_projected[::step, ::step, 0]
    v_vis = velocities_projected[::step, ::step, 1]
    
    # 只在非圆柱体区域绘制箭头
    plt.quiver(X_vis[mask_vis], Y_vis[mask_vis],
               u_vis[mask_vis], v_vis[mask_vis],
               color="dimgray", scale=50.0, width=0.001)
    
    plt.xlim(0, DOMAIN_SIZE_X)
    plt.ylim(0, DOMAIN_SIZE_Y)
    # 设置坐标轴等比例，确保圆形显示为圆形而不是椭圆
    plt.gca().set_aspect('equal')
    plt.title(f'Kármán Vortex Street Simulation (t={time_current:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # clear_output(wait=True)
    # display(plt.gcf())
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
