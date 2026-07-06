from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.gridworld import GridWorld


def collect_trajectory_2d(env, steps=300, normalize=True):
    states = []
    actions = []

    state = env.reset()
    states.append(state)

    for _ in range(steps):
        action = np.random.choice(4)
        next_state = env.step(action)

        states.append(next_state)
        actions.append(action)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)

    if normalize:
        states = states / float(env.size)

    return states, actions


def main():
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = GridWorld(size=10)
    states, actions = collect_trajectory_2d(env, steps=300, normalize=True)

    np.save(output_dir / "trajectories_2d.npy", states)
    np.save(output_dir / "actions_2d.npy", actions)
    print(f"Saved {output_dir / 'trajectories_2d.npy'} with shape: {states.shape}")
    print(f"Saved {output_dir / 'actions_2d.npy'} with shape: {actions.shape}")


if __name__ == "__main__":
    main()
