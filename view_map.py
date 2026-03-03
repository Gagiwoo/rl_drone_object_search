from drone_grid_env import DroneGridEnv
import matplotlib.pyplot as plt
import numpy as np

env = DroneGridEnv(config_file="experiments/distributions/env_config_medium.yaml")

positions = []
for i in range(10):
    obs, info = env.reset()
    obj_map = env.world.object_map
    pos = np.argwhere(obj_map > 0)
    positions.extend(pos)
    print(f"Episode {i}: {len(pos)} objects")

positions = np.array(positions)
plt.figure(figsize=(10, 10))
plt.scatter(positions[:, 1], positions[:, 0], alpha=0.5, s=10)
plt.title("Medium Environment - Should see 4 clusters!")
plt.xlim(0, 48)
plt.ylim(0, 48)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.savefig("medium_env_distribution.png", dpi=150)
plt.show()