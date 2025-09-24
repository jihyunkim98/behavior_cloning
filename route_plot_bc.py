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
    # 가능하면 속도 크기 사용 (Car.vy가 있으면 사용)
    if "Car.vy" in gt_df.columns:
        vy = gt_df["Car.vy"].values
        v_mag = np.sqrt(vx**2 + vy**2)
    else:
        v_mag = vx

    steer = bc_df["pred_steering_angle"].values
    # 초기 상태는 GT의 시작 포즈 (헤딩은 GT 궤적의 접선으로 산출 → 좌표계 일치 보장)
    x0 = gt_df["Car.Road.tx"].iloc[0]
    y0 = gt_df["Car.Road.ty"].iloc[0]
    if len(gt_df) >= 2:
        dx0 = float(gt_df["Car.Road.tx"].iloc[1] - x0)
        dy0 = float(gt_df["Car.Road.ty"].iloc[1] - y0)
        yaw0 = float(np.arctan2(dy0, dx0))
    else:
        yaw_series = gt_df["Car.Yaw"].values
        if np.nanmax(np.abs(yaw_series)) > 3.5:
            yaw_series = np.deg2rad(yaw_series)
        yaw0 = float(yaw_series[0])

    # 파라미터 (근사)
    L = 2.8  # wheelbase [m]
    steer_ratio = 16.0  # steering wheel : road wheel
    max_wheel_deg = 35.0  # 물리적 한계 근사
    alpha = 0.2  # steering smoothing (0: no smooth, 1: overwrite)

    RIGHT_IS_POSITIVE = False  # 음(-) 조향 = 우측

    def rollout_with_combo(sign: float, yaw_offset: float, steps: int):
        tx = [x0]
        ty = [y0]
        yaw = yaw0
        prev_delta_wheel = 0.0
        for i in range(1, steps):
            dt = max(1e-3, float(time[i] - time[i-1]))
            v = float(v_mag[i])
            delta_cmd = float(steer[i])
            # 규약: 양(+)을 우측으로 강제 → 수학 좌표계(CCW 양)와 반대이므로 부호 반전
            if RIGHT_IS_POSITIVE:
                delta_cmd = -delta_cmd
            if abs(delta_cmd) > 3.5:
                delta_cmd = np.deg2rad(delta_cmd)
            delta_wheel = (delta_cmd / steer_ratio) * sign
            delta_wheel = alpha * delta_wheel + (1 - alpha) * prev_delta_wheel
            delta_wheel = np.clip(delta_wheel, -np.deg2rad(max_wheel_deg), np.deg2rad(max_wheel_deg))
            prev_delta_wheel = delta_wheel
            yaw_rate = v / L * np.tan(delta_wheel)
            yaw = yaw + yaw_rate * dt
            if yaw > np.pi:
                yaw -= 2 * np.pi
            elif yaw < -np.pi:
                yaw += 2 * np.pi
            yaw_eff = yaw + yaw_offset
            tx.append(tx[-1] + v * np.cos(yaw_eff) * dt)
            ty.append(ty[-1] + v * np.sin(yaw_eff) * dt)
        return np.asarray(tx), np.asarray(ty)

    # 0) 조향 부호 힌트: GT 곡률변화 방향과의 상관으로 추정 (정보 출력용)
    gx = gt_df["Car.Road.tx"].values
    gy = gt_df["Car.Road.ty"].values
    # 헤딩을 궤적 접선으로 계산해 시간 미분 → yaw_rate_gt
    yaw_path = np.arctan2(np.gradient(gy), np.gradient(gx))
    yaw_unwrap = np.unwrap(yaw_path)
    dt_arr = np.clip(np.gradient(time), 1e-3, None)
    yaw_rate_gt = np.gradient(yaw_unwrap) / dt_arr
    # steering 시리즈(rad, road wheel)
    steer_cmd = steer.astype(float)
    if np.nanmax(np.abs(steer_cmd)) > 3.5:
        steer_cmd = np.deg2rad(steer_cmd)
    wheel_angle = steer_cmd / steer_ratio
    curv_series = np.tan(wheel_angle)
    # 같은 길이로 맞추고 직진구간 제외
    m = min(len(curv_series), len(yaw_rate_gt), len(v_mag))
    curv_series = curv_series[:m]
    yaw_rate_gt = yaw_rate_gt[:m]
    v_for_corr = v_mag[:m]
    mask = np.abs(yaw_rate_gt) > 0.05  # rad/s 임계값
    if mask.any():
        a = np.corrcoef(v_for_corr[mask] * curv_series[mask], yaw_rate_gt[mask])[0, 1]
        b = np.corrcoef(v_for_corr[mask] * (-curv_series[mask]), yaw_rate_gt[mask])[0, 1]
        print(f"[debug] corr(+): {a:.3f}, corr(-): {b:.3f}")
    else:
        print("[debug] insufficient turning to compute correlation; skipping")

    # 1) 짧은 구간으로 sign/yaw_offset/mirror 자동 보정
    calib_steps = min(200, len(time))
    cand = []
    offset_candidates = (0.0, np.pi/2, -np.pi/2, np.pi)
    mirror_candidates = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    def apply_mirror(tx, ty, mx, my):
        # 시작점 (x0,y0)를 기준으로 축 반전
        return x0 + mx * (tx - x0), y0 + my * (ty - y0)
    # 부호는 RIGHT_IS_POSITIVE 규약에 따라 사전 반영되므로 1.0만 시도
    for sgn in (1.0,):
        for off in offset_candidates:
            tx_c, ty_c = rollout_with_combo(sgn, off, calib_steps)
            gx_c = gt_df["Car.Road.tx"].values[:calib_steps]
            gy_c = gt_df["Car.Road.ty"].values[:calib_steps]
            for mx, my in mirror_candidates:
                tx_m, ty_m = apply_mirror(tx_c, ty_c, mx, my)
                err = np.mean(np.hypot(tx_m - gx_c, ty_m - gy_c))
                cand.append((err, sgn, off, mx, my))
    _, best_sign, best_offset, best_mx, best_my = min(cand, key=lambda x: x[0])
    print(f"[debug] chosen offset={best_offset:.3f} rad, mirror=({best_mx:+.0f},{best_my:+.0f})")

    # 최종 전체 롤아웃
    bc_x, bc_y = rollout_with_combo(best_sign, best_offset, len(time))
    bc_x, bc_y = apply_mirror(bc_x, bc_y, best_mx, best_my)
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
