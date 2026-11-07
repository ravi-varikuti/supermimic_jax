"""Convert G1 LAFAN1 CSVs (30 Hz, xyzw) to NPZ (50 Hz, MuJoCo wxyz).

Stores qpos (T, 36), qvel (T, 35), timestamps, FPS, and joint names.
qvel contains world linear velocity, base-local angular velocity, joint velocity.
"""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).parent
INPUT = ROOT / "LAFAN1_Retargeting_Dataset/g1"
OUTPUT = ROOT / "LAFAN1_Retargeting_Dataset/g1_npz"
SOURCE_FPS, FPS = 30, 50

CSV_JOINTS = tuple(
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
)


def convert(path, model):
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid G1 motion: {path}")
    source_time = np.arange(len(raw)) / SOURCE_FPS
    time = np.arange(int(np.floor(source_time[-1] * FPS + 1e-9)) + 1) / FPS
    time = np.minimum(time, source_time[-1])
    qpos = np.empty((len(time), model.nq))
    for column in range(3):
        qpos[:, column] = np.interp(time, source_time, raw[:, column])
    rotations = Slerp(source_time, Rotation.from_quat(raw[:, 3:7]))(time)
    qpos[:, 3:7] = rotations.as_quat()[:, [3, 0, 1, 2]]
    for column, name in enumerate(CSV_JOINTS, start=7):
        address = model.joint(name).qposadr[0]
        qpos[:, address] = np.interp(time, source_time, raw[:, column])

    # Central differences, with one-sided differences at the endpoints.
    qvel = np.empty((len(time), model.nv))
    before = np.maximum(np.arange(len(time)) - 1, 0)
    after = np.minimum(np.arange(len(time)) + 1, len(time) - 1)
    for i, (a, b) in enumerate(zip(before, after)):
        mujoco.mj_differentiatePos(model, qvel[i], time[b] - time[a], qpos[a], qpos[b])
    qvel[:, 3:6] = rotations.inv().apply(rotations[before].apply(qvel[:, 3:6]))

    names = [model.joint(i).name for i in range(1, model.njnt)]
    np.savez_compressed(
        OUTPUT / f"{path.stem}.npz", qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32), time=time, fps=FPS, source_fps=SOURCE_FPS,
        joint_names=np.asarray(names), quaternion_order="wxyz", source_file=path.name,
    )
    return len(time)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(ROOT / "unitree_g1/scene_29dof_mjlab.xml"))
    files = sorted(INPUT.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No G1 CSVs in {INPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in files:
        count = convert(path, model)
        total += count
        print(f"{path.stem}: {count} frames", flush=True)
    print(f"Converted {len(files)} motions, {total} frames -> {OUTPUT}")
