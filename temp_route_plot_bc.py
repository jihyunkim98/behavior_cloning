import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 데이터 경로
# ==========================
map_path = r"C:\Users\BM\Desktop\behavior_cloning\MapData\G29_Test.csv"
gt_path  = r"C:\Users\BM\Desktop\behavior_cloning\DriveData\BC_Track1\BC_G29_Test_142621.csv"
bc_path  = r"C:\Users\BM\Desktop\behavior_cloning\predictions.csv"

# ==========================
# 데이터 로드
# ==========================
map_df = pd.read_csv(map_path)
gt_df  = pd.read_csv(gt_path)
bc_df  = pd.read_csv(bc_path)

map_x, map_y = map_df["SDC.x"].values, map_df["SDC.y"].values
gt_x, gt_y   = gt_df["Car.Road.tx"].values, gt_df["Car.Road.ty"].values

# ==========================
# BC 궤적을 예측 조향으로 적분해 생성 (GT 위치 사용 금지)
# - 속도: GT의 Car.vx 사용 (가속 모델이 없으므로 단순화)
# - 조향: predictions.csv의 pred_steering_angle 사용
# - 포즈 적분: 자전거 모델 근사 (yaw_rate = v/L * tan(steer))
# ==========================
if all(col in bc_df.columns for col in ["pred_steering_angle"]) and \
   all(col in gt_df.columns for col in ["Time", "Car.vx", "Car.Yaw", "Car.Road.tx", "Car.Road.ty"]):
    time = gt_df["Time"].values
    vx = gt_df["Car.vx"].values
    steer = bc_df["pred_steering_angle"].values
    # 초기 상태는 GT의 시작 포즈
    x0 = gt_df["Car.Road.tx"].iloc[0]
    y0 = gt_df["Car.Road.ty"].iloc[0]
    yaw0 = gt_df["Car.Yaw"].iloc[0]

    # 파라미터 (근사)
    L = 2.8  # wheelbase [m]

    bc_x = [x0]
    bc_y = [y0]
    yaw = yaw0
    for i in range(1, len(time)):
        dt = max(1e-3, float(time[i] - time[i-1]))
        v = float(vx[i])
        delta = float(steer[i])
        yaw_rate = v / L * np.tan(delta)
        yaw = yaw + yaw_rate * dt
        x_next = bc_x[-1] + v * np.cos(yaw) * dt
        y_next = bc_y[-1] + v * np.sin(yaw) * dt
        bc_x.append(x_next)
        bc_y.append(y_next)
    bc_x = np.asarray(bc_x)
    bc_y = np.asarray(bc_y)
else:
    # 컬럼이 없으면 이전 방식(GT 위치 재사용)으로 폴백
    bc_x = bc_df["Car.Road.tx"].values
    bc_y = bc_df["Car.Road.ty"].values

N = min(len(gt_x), len(bc_x))

# ==========================
# Figure 설정
# ==========================
plt.ion()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(map_x, map_y, 'k--', alpha=0.5, label="Map Centerline (SDC)")
(gt_traj,) = ax.plot([], [], 'b-', lw=2, label="GT Trajectory")
(bc_traj,) = ax.plot([], [], 'r--', lw=1, label="BC Trajectory")
(gt_dot,)  = ax.plot([], [], 'bo', ms=5, label="GT")
(bc_dot,)  = ax.plot([], [], 'ro', ms=3, label="BC")
ax.set_aspect("equal", adjustable="box")
ax.legend(); ax.grid(True)

# ==========================
# 루프 (빠른 재생)
# ==========================
step = 10       
pause_time = 0.001  

for i in range(1, N, step):
    # 누적 궤적 업데이트
    gt_traj.set_data(gt_x[:i], gt_y[:i])
    bc_traj.set_data(bc_x[:i], bc_y[:i])
    # 현재 위치 점
    gt_dot.set_data([gt_x[i]], [gt_y[i]])
    bc_dot.set_data([bc_x[i]], [bc_y[i]])
    plt.pause(pause_time)

plt.ioff()
plt.show()
