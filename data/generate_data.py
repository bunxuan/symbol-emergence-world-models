"""Generate training trajectories for the symbol emergence experiment."""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "data" / "env_config.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "trajectories.npy"


def generate_1d_trajectory(length=1000, step_size=0.03, x0=0.5, xmin=0.0, xmax=1.0):
    """
    生成一个 1D 小球在 [xmin, xmax] 区间内来回反弹的轨迹。
    返回 shape = (length, 1) 的 numpy 数组。
    """

    x = x0
    v = step_size  # 初始向右

    traj = []

    for _ in range(length):
        traj.append([x])

        # 更新位置
        x_new = x + v

        # 反弹逻辑
        if x_new > xmax:
            x_new = xmax - (x_new - xmax)
            v = -v
        elif x_new < xmin:
            x_new = xmin + (xmin - x_new)
            v = -v

        x = x_new

    return np.array(traj)


def load_config(config_path=CONFIG_PATH):
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    config = load_config()
    traj = generate_1d_trajectory(
        length=config.get("trajectory_length", 1000),
        step_size=config.get("step_size", 0.03),
        x0=config.get("x0", 0.5),
        xmin=config.get("xmin", 0.0),
        xmax=config.get("xmax", 1.0),
    )
    np.save(OUTPUT_PATH, traj)
    print("Saved", OUTPUT_PATH, "with shape:", traj.shape)
