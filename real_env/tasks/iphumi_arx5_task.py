import sys
import time
import os

import numpy as np
import numpy.typing as npt
import hydra
import json
from omegaconf import OmegaConf

from real_env.tasks.base_task import TaskControlMode
from real_env.tasks.umi_arx5_task import UmiARX5Task
from real_env.utils.umi_utils import get_downsampled_camera_images
from robot_utils.pose_utils import get_relative_poses


class iPhumiARX5Task(UmiARX5Task):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_observations(self):
        images_dict_THWC_RGB, camera_timestamps = get_downsampled_camera_images(
            self.wrist_camera_client,
            self.policy_agent.image_history_len,
            self.policy_agent.agent_update_freq_hz,
            cam_obs_freq_hz=self.policy_agent.cam_obs_freq_hz,
        )
        main_camera_timestamps = camera_timestamps["main"]

        eef_poses_list: list[npt.NDArray[np.float64]] = []
        gripper_widths_list: list[npt.NDArray[np.float64]] = []
        print(f"main_camera_timestamps: {main_camera_timestamps}")

        if self.match_camera_latency:
            last_timestamp = main_camera_timestamps[-1]
        else:
            last_timestamp = time.monotonic()
        robot_state_timestamps = [
            last_timestamp - i * (1.0 / self.policy_agent.agent_update_freq_hz)
            for i in range(self.policy_agent.proprio_history_len - 1, -1, -1)
        ]

        for robot_state_timestamp in robot_state_timestamps:
            eef_poses_list.append(
                self.arx5_client.get_eef_xyz_wxyz(timestamp=robot_state_timestamp)
            )
            gripper_widths_list.append(
                self.arx5_client.get_joint_pos(timestamp=robot_state_timestamp)
            )

        eef_poses = np.array(eef_poses_list)  # (N, 7)
        gripper_widths = np.array(gripper_widths_list)  # (N, 1)

        observations = {
            "robot0_eef_xyz_wxyz": eef_poses, # (proprio_history_len, 7)
            "robot0_gripper_width": gripper_widths, # (proprio_history_len, 1)
            "robot0_main_camera": images_dict_THWC_RGB["main"],  # (img_horizon, H, W, 3)
            "robot0_ultrawide_camera": images_dict_THWC_RGB["ultrawide"],  # (img_horizon, H, W, 3)
            # "robot0_depth_camera": images_dict_THWC_RGB["depth"],  # (img_horizon, H, W, 3)
            "timestamps": main_camera_timestamps,  # (img_horizon, )
            "task_name": self.task_name,
        }

        if self.policy_start_eef_xyz_wxyz is None and self.control_mode in (
            TaskControlMode.POLICY,
            TaskControlMode.POLICY_SINGLE_STEP,
        ):
            self.policy_start_eef_xyz_wxyz = eef_poses[0]

        start_pose = (
            self.policy_start_eef_xyz_wxyz
            if self.policy_start_eef_xyz_wxyz is not None
            else eef_poses[0]
        )
        observations["robot0_eef_xyz_wxyz_wrt_start"] = get_relative_poses(
            eef_poses, start_pose
        )

        return observations


def run_iphumi_arx5_task():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    assert len(sys.argv) == 2, "Usage: python run_iphumi_arx5_task.py <task_name>"
    task_name = sys.argv[1]
    with hydra.initialize(config_path="../configs/tasks"):
        cfg = hydra.compose(config_name=f"iphumi_arx5_{task_name}")
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    print(cfg)
    task: iPhumiARX5Task = hydra.utils.instantiate(cfg)

    task.run()


if __name__ == "__main__":
    run_iphumi_arx5_task()
