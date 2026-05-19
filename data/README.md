# Data

This folder stores the 1D environment trajectory used by the world model.

Files:

- `trajectories.npy`: generated trajectory with shape `(T, 1)`
- `env_config.json`: environment and generation settings

The trajectory can be regenerated with:

```bash
python data/generate_data.py
```
