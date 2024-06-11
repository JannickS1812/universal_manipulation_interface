import hydra
import matplotlib.pyplot as plt
import zarr
import pathlib
import numpy as np  # Add import for numpy
from omegaconf import OmegaConf
from diffusion_policy.common.replay_buffer import ReplayBuffer

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(dataset)

    with zarr.ZipStore(cfg.task.dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store,
            store=zarr.MemoryStore()
        )
    eef_pos = np.asarray(replay_buffer['robot0_eef_pos'])
    eef_gripper_width = np.asarray(replay_buffer['robot0_gripper_width'])

    # Plotting the 3D scatter plot
    fig = plt.figure(figsize=(12, 6))

    # 3D scatter plot of end effector positions
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2], c='b')#, marker='o')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D Scatter Plot of End Effector Positions')

    # 2D plot of gripper width over time
    ax2 = fig.add_subplot(122)
    ax2.plot(eef_gripper_width, c='r')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Gripper Width')
    ax2.set_title('Gripper Width Over Time')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
