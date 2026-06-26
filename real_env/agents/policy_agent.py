import copy
import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize
from loguru import logger

from real_env.agents.base_agent import BaseAgent
from robot_utils.pose_utils import (
    get_absolute_pose,
    get_relative_pose,
    pose_9d_to_xyz_wxyz,
    xyz_wxyz_to_pose_9d,
)


class PolicyAgent(BaseAgent):
    def __init__(
        self,
        policy_server_endpoint: str,
        action_as_proprio: bool,
        keep_one_step_dummy_proprio: bool,
        use_relative_action: bool,
        smooth_action_window_size: int,
        **kwargs,
    ):
        """
        keep_one_step_dummy_proprio: For original UMI checkpoints, the last step of proprio pose is actually identity
        smooth_action_window_size: The window size for smoothing the actions. 1 means no smoothing.
        """

        self.policy_rmq_client = RMQClient(
            client_name=f"{kwargs['name']}_policy",
            server_endpoint=policy_server_endpoint,
        )

        logger.info(f"Requesting policy config from {policy_server_endpoint}")

        reply = self.policy_rmq_client.request_with_data(
            topic="policy_config", data=serialize(True), timeout_s=5.0
        )
        self.config: dict[str, Any] = deserialize(reply)
        if "proprio_history_len" not in kwargs:
            kwargs["proprio_history_len"] = (
                self.config["workspace"]["proprio_length"] + 1
            )
            print(f"Using proprio_history_len: {kwargs['proprio_history_len']}")
        if "action_history_len" not in kwargs:
            kwargs["action_history_len"] = (
                self.config["workspace"]["proprio_length"] + 1
            )
            print(f"Using action_history_len: {kwargs['action_history_len']}")
        img_obs_horizon = self.config["workspace"].get("img_obs_horizon")
        if img_obs_horizon is not None:
            kwargs["image_history_len"] = img_obs_horizon
            print(f"Using image_history_len: {kwargs['image_history_len']}")
        super().__init__(**kwargs)
        self.action_as_proprio: bool = action_as_proprio
        self.keep_one_step_dummy_proprio: bool = keep_one_step_dummy_proprio
        self.use_relative_action: bool = use_relative_action
        if self.keep_one_step_dummy_proprio:
            assert (
                self.use_relative_action
            ), "PolicyAgent with keep_one_step_dummy_proprio must use relative action"
        logger.info(
            f"time: {self.config['date_str']}-{self.config['time_str']}, policy_name: {self.config['policy_name']}, run_name: {self.config['run_name']}"
        )

        assert (
            smooth_action_window_size >= 1
        ), "Smooth action window size must be at least 1"
        assert (
            smooth_action_window_size % 2 == 1
        ), "Smooth action window size must be odd"
        self.smooth_action_window_size: int = smooth_action_window_size

    def get_config(self) -> dict[str, Any]:
        return copy.deepcopy(self.config)

    @property
    def cam_obs_freq_hz(self) -> dict[str, float]:
        """Per-camera target observation frequency for deployment.

        Scales agent_update_freq_hz by each camera's training obs rate relative
        to the main camera's training obs rate, so the number of action steps
        between obs frames matches what the policy was trained on.

        Main camera data collected at 60Hz, ultrawide at 10Hz (hardcoded).
        """
        workspace = self.config["workspace"]
        main_training_obs_rate = 60.0 / workspace["obs_down_sample_steps"]
        result = {"main": self.agent_update_freq_hz}
        ultrawide_down_sample_steps = workspace.get("ultrawide_down_sample_steps")
        if ultrawide_down_sample_steps is not None:
            ultrawide_training_obs_rate = 10.0 / ultrawide_down_sample_steps
            result["ultrawide"] = (
                self.agent_update_freq_hz * ultrawide_training_obs_rate / main_training_obs_rate
            )
        return result

    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        observations (per robot i in range(robot_num)):
            robot{i}_eef_xyz_wxyz:              (N, 7)  absolute poses
            robot{i}_gripper_width:             (N, 1)
            robot{i}_main_camera:               (T, H, W, 3) uint8
            robot{i}_ultrawide_camera:          (T, H, W, 3) uint8
            robot{i}_eef_xyz_wxyz_wrt_robot{j}: (N, 7)  cross-arm relative (optional)
            robot{i}_eef_xyz_wxyz_wrt_start:    (N, 7)  episode-start relative (optional)
            timestamps:                         (N,)

        history_actions (per robot i):
            action{i}_eef_xyz_wxyz:  (M, 7)
            action{i}_gripper_width: (M, 1)
            timestamps:              (M,)

        Returns absolute poses:
            action{i}_eef_xyz_wxyz:  (K, 7)
            action{i}_gripper_width: (K, 1)
            timestamps:              (K,)
        """
        observations = copy.deepcopy(observations)
        history_actions = copy.deepcopy(history_actions)

        obs_last_timestamp: npt.NDArray[np.float64] = observations["timestamps"][-1]

        # Save actual EEF poses before action_as_proprio substitution —
        # used to reconstruct absolute actions from relative outputs.
        obs_last_eef: list[npt.NDArray[np.float64]] = [
            observations[f"robot{i}_eef_xyz_wxyz"][-1].copy()
            for i in range(self.robot_num)
        ]

        # Replace per-robot EEF/gripper obs with commanded action history.
        if self.action_as_proprio:
            for i in range(self.robot_num):
                if f"action{i}_eef_xyz_wxyz" in history_actions:
                    observations[f"robot{i}_eef_xyz_wxyz"] = history_actions[f"action{i}_eef_xyz_wxyz"]
                    observations[f"robot{i}_gripper_width"] = history_actions[f"action{i}_gripper_width"]

        actual_obs_length = len(observations["robot0_eef_xyz_wxyz"])

        obs_to_send: dict[str, npt.NDArray] = {}

        if self.use_relative_action:
            if not self.keep_one_step_dummy_proprio:
                actual_obs_length -= 1
                for i in range(self.robot_num):
                    observations[f"robot{i}_gripper_width"] = observations[f"robot{i}_gripper_width"][1:]

            for i in range(self.robot_num):
                poses = observations[f"robot{i}_eef_xyz_wxyz"]
                ref = poses[-1]
                rel = np.array([
                    get_relative_pose(poses[t], ref)
                    for t in range(actual_obs_length)
                ], dtype=np.float64)
                pose9d = xyz_wxyz_to_pose_9d(rel)
                obs_to_send[f"robot{i}_eef_pos"] = pose9d[:, :3].astype(np.float32)
                obs_to_send[f"robot{i}_eef_rot_axis_angle"] = pose9d[:, 3:].astype(np.float32)
        else:
            for i in range(self.robot_num):
                poses = observations[f"robot{i}_eef_xyz_wxyz"][:actual_obs_length]
                pose9d = xyz_wxyz_to_pose_9d(poses.astype(np.float64))
                obs_to_send[f"robot{i}_eef_pos"] = pose9d[:, :3].astype(np.float32)
                obs_to_send[f"robot{i}_eef_rot_axis_angle"] = pose9d[:, 3:].astype(np.float32)

        for i in range(self.robot_num):
            obs_to_send[f"robot{i}_gripper_width"] = (
                observations[f"robot{i}_gripper_width"][:actual_obs_length].astype(np.float32)
            )

        # Cross-arm obs: task pre-computes as xyz_wxyz; convert to pos3+rot6d.
        for i in range(self.robot_num):
            for j in range(self.robot_num):
                if i == j:
                    continue
                key = f"robot{i}_eef_xyz_wxyz_wrt_robot{j}"
                if key in observations:
                    pose9d = xyz_wxyz_to_pose_9d(observations[key][:actual_obs_length].astype(np.float64))
                    obs_to_send[f"robot{i}_eef_pos_wrt_robot{j}"] = pose9d[:, :3].astype(np.float32)
                    obs_to_send[f"robot{i}_eef_rot_axis_angle_wrt_robot{j}"] = pose9d[:, 3:].astype(np.float32)

        # Wrt-start obs: only rotation used by model.
        for i in range(self.robot_num):
            key = f"robot{i}_eef_xyz_wxyz_wrt_start"
            if key in observations:
                pose9d = xyz_wxyz_to_pose_9d(observations[key].astype(np.float64))
                obs_to_send[f"robot{i}_eef_rot_axis_angle_wrt_start"] = (
                    pose9d[:actual_obs_length, 3:].astype(np.float32)
                )

        # Camera images and task_name pass through unchanged.
        for key, val in observations.items():
            if "camera" in key or key == "task_name":
                obs_to_send[key] = val

        obs_to_send["episode_idx"] = np.array(0)

        reply = self.policy_rmq_client.request_with_data(
            topic="policy_inference", data=serialize(obs_to_send), timeout_s=5.0
        )

        raw_actions = deserialize(reply)
        if isinstance(raw_actions, str):
            raise ValueError(f"Policy agent returned an error message: {raw_actions}")
        assert isinstance(raw_actions, dict)

        actions: dict[str, npt.NDArray[np.float64]] = {}

        for i in range(self.robot_num):
            raw_pos = np.array(raw_actions[f"action{i}_eef_pos"])            # (K, 3)
            raw_rot = np.array(raw_actions[f"action{i}_eef_rot_axis_angle"]) # (K, 6)
            rel_xyz_wxyz = pose_9d_to_xyz_wxyz(
                np.concatenate([raw_pos, raw_rot], axis=-1).astype(np.float64)
            )  # (K, 7)

            if self.use_relative_action:
                absolute_eef = np.array([
                    get_absolute_pose(obs_last_eef[i], rel_xyz_wxyz[k])
                    for k in range(len(rel_xyz_wxyz))
                ])
            else:
                absolute_eef = rel_xyz_wxyz

            if self.smooth_action_window_size > 1:
                for axis in range(3):
                    side_length = (self.smooth_action_window_size - 1) // 2
                    absolute_eef[side_length:-side_length, axis] = np.convolve(
                        absolute_eef[:, axis],
                        np.ones(self.smooth_action_window_size) / self.smooth_action_window_size,
                        mode="valid",
                    )
                    for timestep in range(1, side_length):
                        absolute_eef[timestep, axis] = np.mean(absolute_eef[: timestep * 2 + 1, axis])
                        absolute_eef[-timestep - 1, axis] = np.mean(absolute_eef[-timestep * 2 - 1 :, axis])

            actions[f"action{i}_eef_xyz_wxyz"] = absolute_eef
            actions[f"action{i}_gripper_width"] = np.array(raw_actions[f"action{i}_gripper_width"])

        actions["timestamps"] = (
            obs_last_timestamp
            + np.arange(0, self.action_prediction_horizon)
            / self.agent_update_freq_hz
        )

        return actions

    def reset(self):
        reply = self.policy_rmq_client.request_with_data(
            topic="policy_reset", data=serialize("RESET"), timeout_s=5.0
        )
        reply_str = deserialize(reply)

        if isinstance(reply_str, str):
            assert reply_str == "OK", f"Policy agent returned an error: {reply_str}"
        elif isinstance(reply_str, bool):
            assert reply_str, f"Policy agent returned an error: {reply_str}"
        else:
            raise ValueError(
                f"Policy agent returned an unexpected type: {type(reply_str)}"
            )

    def uses_prompting(self) -> bool:
        return bool(self.config.get("use_prompting", False))

    def send_prompt_request(self, task_name: str, prompt_index: int) -> None:
        """Ask the policy server to load and apply a behavior prompt by task name and index."""
        reply = self.policy_rmq_client.request_with_data(
            topic="policy_prompt",
            data=serialize({"task_name": task_name, "prompt_index": prompt_index}),
            timeout_s=60.0,
        )
        reply_str = deserialize(reply)
        if isinstance(reply_str, str):
            assert reply_str == "OK", f"Policy prompt error: {reply_str}"
        elif isinstance(reply_str, bool):
            assert reply_str, f"Policy prompt error: {reply_str}"
        else:
            raise ValueError(f"Policy prompt returned unexpected type: {type(reply_str)}")

    def export_data(self):
        time_str = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # YYYY-MM-DD_HH-MM-SS
        reply = deserialize(
            self.policy_rmq_client.request_with_data(
                topic="export_recorded_data", data=serialize(time_str), timeout_s=5.0
            )
        )

        print(f"Policy agent exported data to: {reply}")
