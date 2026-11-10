"""Default G1 dimensions for the current environment."""

from robot_config import RobotConfig
from dataclasses import replace
import re
import numpy as np


G1_LEGACY_CONFIG = RobotConfig(
    state_shapes=(("qpos", (36,)), ("qvel", (35,))),
    context_shapes=(
        ("friction", ()),
        ("joint_bias", (29,)),
        ("torso_com_offset", (3,)),
    ),
    action_dim=29,
    observation_dim=93,
    history_length=10,
    quaternion_slices=((3, 7),),
)

G1_CONFIG = replace(G1_LEGACY_CONFIG, observation_dim=99)


def controller_defaults(model):
    """Named nominal pose and motor-scaled position offsets, independent of qpos0."""
    pose = {".*_hip_pitch_joint": -.312, ".*_knee_joint": .669,
            ".*_ankle_pitch_joint": -.363, ".*_elbow_joint": .6,
            "left_shoulder_roll_joint": .2, "right_shoulder_roll_joint": -.2,
            ".*_shoulder_pitch_joint": .2}
    joints = model.actuator_trnid[:, 0]
    nominal = np.array([next((v for pattern, v in pose.items()
                              if re.fullmatch(pattern, model.joint(int(j)).name)), 0.) for j in joints], np.float32)
    gains = model.actuator_gainprm[:, 0]
    if np.any(gains <= 0):
        raise ValueError("Motor scaling requires positive position gains")
    scale = .25 * np.max(np.abs(model.actuator_forcerange), axis=-1) / gains
    return nominal, scale.astype(np.float32)
