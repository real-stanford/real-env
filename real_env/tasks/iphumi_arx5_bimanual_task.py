import copy
import json
import os
import sys
import time

import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from real_env.agents.policy_agent import PolicyAgent
from real_env.controllers.base_controller_client import BaseCartesianJointClient
from real_env.peripherals.base_camera import BaseCameraClient
from real_env.tasks.base_task import BaseTask, TaskControlMode
from robologger.classes import Morphology
from real_env.utils.umi_utils import get_downsampled_camera_images
from robot_utils.pose_utils import get_absolute_pose, get_relative_poses
from robot_utils.time_utils import wait_until


class iPhumiARX5BimanualTask(BaseTask):
    """Bimanual iPhUMI task with two ARX5 arms.

    Left arm  ↔  gripper_left_*  ↔  action_left_*
    Right arm ↔  gripper_right_* ↔  action_right_*

    SpaceMouse controls the right arm by default.
    Press 'l' to switch to left arm, ';' to switch back to right arm.
    """

    def __init__(
        self,
        # Left arm
        arx5_left_controller_endpoint: str,
        arx5_left_home_gripper_pos_m: float | None,
        arx5_left_home_pose_xyz_wxyz: list | None,
        wrist_camera_left_endpoint: str,
        # Right arm
        arx5_right_controller_endpoint: str,
        arx5_right_home_gripper_pos_m: float | None,
        arx5_right_home_pose_xyz_wxyz: list | None,
        wrist_camera_right_endpoint: str,
        # Control params
        arx5_lookahead_time_s: float,
        pause_when_policy_inference: bool,
        camera_latency_s: float,
        policy_latency_s: float,
        policy_agent: PolicyAgent,
        run_name: str,
        task_name: str,
        # Prompting
        prompt_index: int = 0,
        match_camera_latency: bool = False,
        # 4x4 SE3 transform from right arm's base frame to left arm's base frame.
        # Used to express cross-arm relative poses in a common frame, matching training.
        tx_left_right_base: list | None = None,
        **kwargs,
    ) -> None:
        print("Initializing iPhumiARX5BimanualTask")

        self.policy_agent: PolicyAgent = policy_agent

        if "<POLICY_NAME>" in run_name:
            run_name = run_name.replace("<POLICY_NAME>", self.policy_agent.config["run_name"])
        if "<POLICY_EPOCH>" in run_name:
            run_name = run_name.replace("<POLICY_EPOCH>", str(self.policy_agent.config["epoch"]))

        self.task_name: str = task_name
        super().__init__(run_name=run_name, morphology=Morphology.BI_MANUAL, task_name=task_name.replace(" ", "_"), **kwargs)

        self.clients.append(self.policy_agent)

        # Left arm
        self.arx5_left_client = BaseCartesianJointClient(
            name="ARX5_left",
            server_endpoint=arx5_left_controller_endpoint,
            data_remaining_time_s=1.0,
        )
        self.clients.append(self.arx5_left_client)
        self.arx5_left_home_gripper_pos_m: npt.NDArray[np.float64] = np.array(
            [arx5_left_home_gripper_pos_m] if arx5_left_home_gripper_pos_m is not None else [0.08]
        )
        self.arx5_left_home_pose_xyz_wxyz: npt.NDArray[np.float64] | None = (
            np.array(arx5_left_home_pose_xyz_wxyz) if arx5_left_home_pose_xyz_wxyz is not None else None
        )

        # Left wrist camera
        self.wrist_camera_left_client = BaseCameraClient(
            name="wrist_camera_left",
            server_endpoint=wrist_camera_left_endpoint,
        )
        self.clients.append(self.wrist_camera_left_client)

        # Right arm
        self.arx5_right_client = BaseCartesianJointClient(
            name="ARX5_right",
            server_endpoint=arx5_right_controller_endpoint,
            data_remaining_time_s=1.0,
        )
        self.clients.append(self.arx5_right_client)
        self.arx5_right_home_gripper_pos_m: npt.NDArray[np.float64] = np.array(
            [arx5_right_home_gripper_pos_m] if arx5_right_home_gripper_pos_m is not None else [0.08]
        )
        self.arx5_right_home_pose_xyz_wxyz: npt.NDArray[np.float64] | None = (
            np.array(arx5_right_home_pose_xyz_wxyz) if arx5_right_home_pose_xyz_wxyz is not None else None
        )

        # Right wrist camera
        self.wrist_camera_right_client = BaseCameraClient(
            name="wrist_camera_right",
            server_endpoint=wrist_camera_right_endpoint,
        )
        self.clients.append(self.wrist_camera_right_client)

        # Control params
        self.arx5_lookahead_time_s: float = arx5_lookahead_time_s
        self.pause_when_policy_inference: bool = pause_when_policy_inference
        self.camera_latency_s: float = camera_latency_s
        self.policy_latency_s: float = policy_latency_s
        self.match_camera_latency: bool = match_camera_latency
        self.tx_left_right_base: npt.NDArray[np.float64] | None = (
            np.array(tx_left_right_base, dtype=np.float64) if tx_left_right_base is not None else None
        )

        # Prompting
        self.prompt_index: int = prompt_index

        # Seed from actual position, not the last commanded target, so relaunching the
        # script doesn't jump the arm back to where it was in the previous session.
        right_eef = self.arx5_right_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        right_gripper = self.arx5_right_client.get_joint_pos(timestamp=time.monotonic())
        self.spacemouse_history_actions: dict[str, npt.NDArray[np.float64]] = {
            "action0_eef_xyz_wxyz": right_eef[np.newaxis, :],
            "action0_gripper_width": right_gripper[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }
        self.policy_history_actions: dict[str, npt.NDArray[np.float64]] = {}
        self.last_control_timestamp: float = time.monotonic()
        self.last_control_mode: TaskControlMode = TaskControlMode.SPACEMOUSE

        # Episode-start poses (tracked during policy execution for start-relative obs)
        self.policy_start_eef_xyz_wxyz_left: npt.NDArray[np.float64] | None = None
        self.policy_start_eef_xyz_wxyz_right: npt.NDArray[np.float64] | None = None

        # SpaceMouse arm selection ('right' by default; press 'l' or ';')
        self.spacemouse_arm: str = "right"
        self._last_spacemouse_arm: str = "right"
        self._aligning: bool = False  # suppresses spacemouse commands during 'o'-key alignment
        self._prompted: bool = False  # set by 'p'; cleared on reset_policy_agent
        self._inactive_held_eef: npt.NDArray[np.float64] | None = None
        self._inactive_held_gripper: npt.NDArray[np.float64] | None = None

        # EEF pose visualizer (toggled with 'v')
        self._pose_vis = None

        self.reset_policy_agent()

    # ------------------------------------------------------------------
    # Extra key handling: 'l' = left arm, ';' = right arm
    # ------------------------------------------------------------------

    def extra_key_help(self) -> list[str]:
        return [
            "l        spacemouse → left arm",
            ";        spacemouse → right arm",
            "p        send prompt",
            "o        align left arm to demo",
            "h        go to home pose",
            "v        toggle pose visualizer",
        ]

    def handle_extra_terminal_key(self, key: str) -> bool:
        if key == "l":
            self.spacemouse_arm = "left"
            print("SpaceMouse → LEFT arm")
            return True
        elif key == ";":
            self.spacemouse_arm = "right"
            print("SpaceMouse → RIGHT arm")
            return True
        elif key == "p":
            self._send_prompt()
            return True
        elif key == "o":
            self._align_left_arm_to_demo()
            return True
        elif key == "h":
            self._go_to_home_pose()
            return True
        elif key == "v":
            self._toggle_pose_visualizer()
            return True
        return False

    def _toggle_pose_visualizer(self) -> None:
        """Toggle the live Open3D EEF pose visualizer (press 'v')."""
        from real_env.utils.pose_visualizer import PoseVisualizer

        if self._pose_vis is not None and self._pose_vis.is_running:
            self._pose_vis.stop()
            self._pose_vis = None
            return

        if self.tx_left_right_base is None:
            print("[PoseVis] Cannot start: tx_left_right_base not configured.")
            return

        self._pose_vis = PoseVisualizer(self.tx_left_right_base)
        self._pose_vis.start()

    def _update_pose_vis(
        self,
        left_xyz_wxyz: npt.NDArray[np.float64],
        right_xyz_wxyz: npt.NDArray[np.float64],
        left_wrt_start_xyz_wxyz: npt.NDArray[np.float64] | None = None,
        right_wrt_start_xyz_wxyz: npt.NDArray[np.float64] | None = None,
        left_wrt_right_xyz_wxyz: npt.NDArray[np.float64] | None = None,
        right_wrt_left_xyz_wxyz: npt.NDArray[np.float64] | None = None,
    ) -> None:
        if self._pose_vis is not None and self._pose_vis.is_running:
            self._pose_vis.update(
                left_xyz_wxyz, right_xyz_wxyz,
                left_wrt_start_xyz_wxyz, right_wrt_start_xyz_wxyz,
                left_wrt_right_xyz_wxyz, right_wrt_left_xyz_wxyz,
            )

    def _go_to_home_pose(self) -> None:
        """Move both arms to arx5_{left,right}_home_pose_xyz_wxyz (press 'h')."""
        if self.arx5_left_home_pose_xyz_wxyz is None and self.arx5_right_home_pose_xyz_wxyz is None:
            print("[home] No home poses configured (arx5_left/right_home_pose_xyz_wxyz are null).")
            return
        print("[home] Moving to home poses ...")
        move_duration_s = 3.0
        dt = 0.1
        self._aligning = True
        start = time.monotonic()
        eef_deadline = start + move_duration_s + self.arx5_lookahead_time_s
        grip_deadline = start + move_duration_s
        left_eef_target = (
            self.arx5_left_home_pose_xyz_wxyz
            if self.arx5_left_home_pose_xyz_wxyz is not None
            else self.arx5_left_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        )
        right_eef_target = (
            self.arx5_right_home_pose_xyz_wxyz
            if self.arx5_right_home_pose_xyz_wxyz is not None
            else self.arx5_right_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        )
        while time.monotonic() - start < move_duration_s:
            self.arx5_left_client.schedule_eef_traj(
                eef_traj_xyz_wxyz=left_eef_target[np.newaxis, :],
                timestamps=np.array([eef_deadline]),
                use_relative_timestamps=False,
            )
            self.arx5_left_client.schedule_joint_traj(
                joint_traj_pos=self.arx5_left_home_gripper_pos_m[np.newaxis, :],
                timestamps=np.array([grip_deadline]),
                use_relative_timestamps=False,
            )
            self.arx5_right_client.schedule_eef_traj(
                eef_traj_xyz_wxyz=right_eef_target[np.newaxis, :],
                timestamps=np.array([eef_deadline]),
                use_relative_timestamps=False,
            )
            self.arx5_right_client.schedule_joint_traj(
                joint_traj_pos=self.arx5_right_home_gripper_pos_m[np.newaxis, :],
                timestamps=np.array([grip_deadline]),
                use_relative_timestamps=False,
            )
            time.sleep(dt)
        active_client = self.arx5_left_client if self.spacemouse_arm == "left" else self.arx5_right_client
        self.spacemouse_history_actions = {
            "action0_eef_xyz_wxyz": active_client.get_eef_xyz_wxyz(timestamp=time.monotonic())[np.newaxis, :],
            "action0_gripper_width": active_client.get_joint_pos(timestamp=time.monotonic())[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }
        self._inactive_held_eef = None  # force reseed from actual post-move position
        self._aligning = False
        print("[home] Done.")

    def _align_left_arm_to_demo(self) -> None:
        """Query the replay server for the demo's initial inter-gripper transform and move
        the left arm to match it.  Right arm stays fixed.  Blocks for ~5 s."""
        from robotmq import deserialize, serialize

        try:
            reply = self.policy_agent.policy_rmq_client.request_with_data(
                topic="query_initial_gripper_transform",
                data=serialize({}),
                timeout_s=5.0,
            )
            result = deserialize(reply)
        except Exception as e:
            print(f"[align] Failed to query replay server: {e}")
            return
        if isinstance(result, str):
            print(f"[align] Replay server error: {result}")
            return

        # transform_xyz_wxyz = left wrt right at demo step 0 (inv(demo_R[0]) @ demo_L[0])
        transform_xyz_wxyz = np.array(result["transform_xyz_wxyz"], dtype=np.float64)  # (7,)

        right_current = self.arx5_right_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        left_current  = self.arx5_left_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        left_gripper_cur = self.arx5_left_client.get_joint_pos(timestamp=time.monotonic())
        right_gripper_cur = self.arx5_right_client.get_joint_pos(timestamp=time.monotonic())

        # Express right arm EEF in left arm base frame (B_R → B_L via tx_left_right_base)
        if self.tx_left_right_base is not None:
            right_in_B_L = self._transform_pose_xyz_wxyz(self.tx_left_right_base, right_current)
        else:
            right_in_B_L = right_current.copy()
            print("[align] WARNING: tx_left_right_base not set — assuming shared base frame")

        # Current left-wrt-right inter-gripper transform (same computation as cross-arm obs)
        current_intergrip = get_relative_poses(left_current[np.newaxis], right_in_B_L)[0]

        print("[align] Demo initial inter-gripper transform (left wrt right, xyz_wxyz):")
        print(f"         {np.round(transform_xyz_wxyz, 4)}")
        print("[align] Current inter-gripper transform (left wrt right, xyz_wxyz):")
        print(f"         {np.round(current_intergrip, 4)}")

        confirm = input("[align] Proceed with left-arm alignment? [y/N] ").strip().lower()
        if confirm != "y":
            print("[align] Aborted.")
            return

        # left_target = (tx @ right_B_R) @ transform  (both in B_L frame)
        # After this move: inv(tx @ right_B_R) @ left_target = transform ✓
        left_target = get_absolute_pose(right_in_B_L, transform_xyz_wxyz)

        print(f"[align] Left arm target (xyz_wxyz): {np.round(left_target, 4)}")
        print(f"[align] Moving left arm — hold right arm still ...")

        # Drive left arm to target over move_duration_s seconds.
        # Use absolute timestamps (use_relative_timestamps=False) so the arm controller
        # always knows it has [deadline - now] seconds to reach the target.  As time
        # passes the remaining window shrinks naturally, causing smooth acceleration —
        # the same approach used by run_policy_control.
        # With relative timestamps of only dt+lookahead (0.2 s), the arm would see an
        # unreachable far-away target and silently do nothing.
        move_duration_s = 2.0
        dt = 0.1
        self._aligning = True
        start = time.monotonic()
        eef_deadline  = start + move_duration_s + self.arx5_lookahead_time_s
        grip_deadline = start + move_duration_s
        while time.monotonic() - start < move_duration_s:
            self.arx5_left_client.schedule_eef_traj(
                eef_traj_xyz_wxyz=left_target[np.newaxis, :],
                timestamps=np.array([eef_deadline]),
                use_relative_timestamps=False,
            )
            self.arx5_left_client.schedule_joint_traj(
                joint_traj_pos=left_gripper_cur[np.newaxis, :],
                timestamps=np.array([grip_deadline]),
                use_relative_timestamps=False,
            )
            self.arx5_right_client.schedule_eef_traj(
                eef_traj_xyz_wxyz=right_current[np.newaxis, :],
                timestamps=np.array([eef_deadline]),
                use_relative_timestamps=False,
            )
            self.arx5_right_client.schedule_joint_traj(
                joint_traj_pos=right_gripper_cur[np.newaxis, :],
                timestamps=np.array([grip_deadline]),
                use_relative_timestamps=False,
            )
            time.sleep(dt)

        # Re-seed spacemouse history from actual arm positions so the spacemouse loop
        # doesn't snap the active arm back to its pre-alignment pose on the next cycle.
        active_client = (
            self.arx5_left_client if self.spacemouse_arm == "left"
            else self.arx5_right_client
        )
        active_eef_now    = active_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        active_gripper_now = active_client.get_joint_pos(timestamp=time.monotonic())
        self.spacemouse_history_actions = {
            "action0_eef_xyz_wxyz": active_eef_now[np.newaxis, :],
            "action0_gripper_width": active_gripper_now[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }

        self._inactive_held_eef = None  # force reseed from actual post-move position
        self._aligning = False
        print("[align] Done. Verify position, then press 'c' to start replay.")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _send_prompt(self) -> None:
        if not self.policy_agent.uses_prompting():
            print("[prompt] Policy was not trained with prompting — ignoring")
            return
        if not self._prompted:
            print(f"[prompt] Requesting prompt: task={self.task_name!r}, idx={self.prompt_index}")
            self.policy_agent.send_prompt_request(self.task_name, self.prompt_index)
            self._prompted = True
            print("[prompt] Done. Press 'c' to start execution.")
        else:
            print("[prompt] Already prompted and not reset since — ignoring")

    def start_episode(self):
        if not self._prompted:
            self._send_prompt()
        return super().start_episode()

    def reset(self):
        self.arx5_left_client.reset()
        self.arx5_right_client.reset()

        self.last_control_timestamp = time.monotonic()
        right_eef = self.arx5_right_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        right_gripper = self.arx5_right_client.get_joint_pos(timestamp=time.monotonic())
        self.spacemouse_history_actions = {
            "action0_eef_xyz_wxyz": right_eef[np.newaxis, :],
            "action0_gripper_width": right_gripper[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }
        self._inactive_held_eef = None
        self._inactive_held_gripper = None

    def reset_policy_agent(self):
        self.policy_agent.reset()
        self.policy_history_actions = {}
        self.policy_start_eef_xyz_wxyz_left = None
        self.policy_start_eef_xyz_wxyz_right = None
        self._prompted = False

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_observations(self) -> dict[str, npt.NDArray]:
        left_images, _ = get_downsampled_camera_images(
            self.wrist_camera_left_client,
            self.policy_agent.image_history_len,
            self.policy_agent.agent_update_freq_hz,
            cam_obs_freq_hz=self.policy_agent.cam_obs_freq_hz,
        )
        right_images, right_camera_timestamps = get_downsampled_camera_images(
            self.wrist_camera_right_client,
            self.policy_agent.image_history_len,
            self.policy_agent.agent_update_freq_hz,
            cam_obs_freq_hz=self.policy_agent.cam_obs_freq_hz,
        )
        right_main_camera_timestamps = right_camera_timestamps["main"]

        if self.match_camera_latency:
            last_timestamp = right_main_camera_timestamps[-1]
        else:
            last_timestamp = time.monotonic()

        robot_state_timestamps = [
            last_timestamp - i * (1.0 / self.policy_agent.agent_update_freq_hz)
            for i in range(self.policy_agent.proprio_history_len - 1, -1, -1)
        ] # 

        left_eef_poses: list[npt.NDArray[np.float64]] = []
        left_gripper_widths: list[npt.NDArray[np.float64]] = []
        right_eef_poses: list[npt.NDArray[np.float64]] = []
        right_gripper_widths: list[npt.NDArray[np.float64]] = []

        for ts in robot_state_timestamps:
            left_eef_poses.append(self.arx5_left_client.get_eef_xyz_wxyz(timestamp=ts))
            left_gripper_widths.append(self.arx5_left_client.get_joint_pos(timestamp=ts))
            right_eef_poses.append(self.arx5_right_client.get_eef_xyz_wxyz(timestamp=ts))
            right_gripper_widths.append(self.arx5_right_client.get_joint_pos(timestamp=ts))

        left_eef_poses_arr = np.array(left_eef_poses)    # (N, 7)
        right_eef_poses_arr = np.array(right_eef_poses)  # (N, 7)

        # robot0 = left arm, robot1 = right arm
        obs = {
            "robot0_eef_xyz_wxyz": left_eef_poses_arr, # absolute; policy agent handles making this relative
            "robot0_gripper_width": np.array(left_gripper_widths),
            "robot0_main_camera": left_images["main"],
            "robot0_ultrawide_camera": left_images["ultrawide"],
            "robot1_eef_xyz_wxyz": right_eef_poses_arr, # absolute; policy agent handles making this relative
            "robot1_gripper_width": np.array(right_gripper_widths),
            "robot1_main_camera": right_images["main"],
            "robot1_ultrawide_camera": right_images["ultrawide"],
            "timestamps": right_main_camera_timestamps,
        }

        # Cross-arm relative obs (task owns the base-frame transforms).
        if self.tx_left_right_base is not None:
            right_last_in_left = self._transform_pose_xyz_wxyz(
                self.tx_left_right_base, right_eef_poses_arr[-1]
            )
            obs["robot0_eef_xyz_wxyz_wrt_robot1"] = get_relative_poses(
                left_eef_poses_arr, right_last_in_left
            )
            left_last_in_right = self._transform_pose_xyz_wxyz(
                np.linalg.inv(self.tx_left_right_base), left_eef_poses_arr[-1]
            )
            obs["robot1_eef_xyz_wxyz_wrt_robot0"] = get_relative_poses(
                right_eef_poses_arr, left_last_in_right
            )

        # Track episode-start poses once policy begins executing.
        if self.control_mode in (TaskControlMode.POLICY, TaskControlMode.POLICY_SINGLE_STEP):
            if self.policy_start_eef_xyz_wxyz_left is None:
                self.policy_start_eef_xyz_wxyz_left = left_eef_poses_arr[0]
                self.policy_start_eef_xyz_wxyz_right = right_eef_poses_arr[0]

        # Always include wrt-start obs; use current pose as reference before episode starts
        # (produces identity relative pose before policy mode).
        start_left = (
            self.policy_start_eef_xyz_wxyz_left
            if self.policy_start_eef_xyz_wxyz_left is not None
            else left_eef_poses_arr[0]
        )
        start_right = (
            self.policy_start_eef_xyz_wxyz_right
            if self.policy_start_eef_xyz_wxyz_right is not None
            else right_eef_poses_arr[0]
        )
        obs["robot0_eef_xyz_wxyz_wrt_start"] = get_relative_poses(left_eef_poses_arr, start_left) # relative
        obs["robot1_eef_xyz_wxyz_wrt_start"] = get_relative_poses(right_eef_poses_arr, start_right) # relative
        obs["task_name"] = self.task_name

        self._update_pose_vis(
            left_eef_poses_arr[-1],
            right_eef_poses_arr[-1],
            left_wrt_start_xyz_wxyz=obs["robot0_eef_xyz_wxyz_wrt_start"][-1],
            right_wrt_start_xyz_wxyz=obs["robot1_eef_xyz_wxyz_wrt_start"][-1],
            left_wrt_right_xyz_wxyz=obs["robot0_eef_xyz_wxyz_wrt_robot1"][-1] if "robot0_eef_xyz_wxyz_wrt_robot1" in obs else None,
            right_wrt_left_xyz_wxyz=obs["robot1_eef_xyz_wxyz_wrt_robot0"][-1] if "robot1_eef_xyz_wxyz_wrt_robot0" in obs else None,
        )

        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_pose_xyz_wxyz(tx: npt.NDArray[np.float64], pose_xyz_wxyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply a 4x4 SE3 matrix to a (7,) xyz_wxyz pose, returning (7,) in the new frame."""
        mat = np.eye(4)
        wxyz = pose_xyz_wxyz[3:]
        mat[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        mat[:3, 3] = pose_xyz_wxyz[:3]
        result = tx @ mat
        xyzw_out = Rotation.from_matrix(result[:3, :3]).as_quat()
        return np.concatenate([result[:3, 3], xyzw_out[[3, 0, 1, 2]]])

    # ------------------------------------------------------------------
    # Policy control
    # ------------------------------------------------------------------

    def run_policy_control(self):
        observations = self.get_observations()

        deployment_actions = self.policy_agent.predict_actions(
            observations=observations,
            history_actions=self.get_history_actions(length=self.policy_agent.action_history_len),
        )

        if (
            self.pause_when_policy_inference
            or self.last_control_mode == TaskControlMode.POLICY_SINGLE_STEP
            or self.last_control_mode == TaskControlMode.SPACEMOUSE
        ):
            deployment_actions["timestamps"] = (
                time.monotonic()
                + np.arange(self.policy_agent.action_prediction_horizon)
                / self.policy_agent.agent_update_freq_hz
            )

        if (
            self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
            or self.pause_when_policy_inference
        ):
            for k in list(deployment_actions.keys()):
                deployment_actions[k] = deployment_actions[k][: self.policy_agent.action_execution_horizon]

        self.arx5_left_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=deployment_actions["action0_eef_xyz_wxyz"],
            timestamps=deployment_actions["timestamps"] + self.arx5_lookahead_time_s,
            use_relative_timestamps=False,
        )
        self.arx5_left_client.schedule_joint_traj(
            joint_traj_pos=deployment_actions["action0_gripper_width"],
            timestamps=deployment_actions["timestamps"],
            use_relative_timestamps=False,
        )
        self.arx5_right_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=deployment_actions["action1_eef_xyz_wxyz"],
            timestamps=deployment_actions["timestamps"] + self.arx5_lookahead_time_s,
            use_relative_timestamps=False,
        )
        self.arx5_right_client.schedule_joint_traj(
            joint_traj_pos=deployment_actions["action1_gripper_width"],
            timestamps=deployment_actions["timestamps"],
            use_relative_timestamps=False,
        )

        # the addition of + 1 here is questionable. Really we do not know how many of the predicted actions were actually executed because the controller does some fancy stitching of the new trajectory. Also if we are using pause_when_policy_inference==True then this + 1 is just definitely wrong.
        self.policy_history_actions = {
            k: v[: min(self.policy_agent.action_execution_horizon + 1, v.shape[0])]
            for k, v in deployment_actions.items()
        }

        wait_end = deployment_actions["timestamps"][self.policy_agent.action_execution_horizon - 1]
        if (
            self.pause_when_policy_inference
            or self.last_control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            wait_until(monotonic_time=wait_end + self.camera_latency_s)
        else:
            wait_until(monotonic_time=wait_end - 0.02)

        self.last_control_mode = self.control_mode

    # ------------------------------------------------------------------
    # SpaceMouse control
    # ------------------------------------------------------------------

    def run_spacemouse_control(self):
        if self._aligning:
            return

        right_eef = self.arx5_right_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        right_gripper = self.arx5_right_client.get_joint_pos(timestamp=time.monotonic())
        left_eef = self.arx5_left_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        left_gripper = self.arx5_left_client.get_joint_pos(timestamp=time.monotonic())

        self._update_pose_vis(left_eef, right_eef)

        # Active arm is selected by 'l' / ';' keys
        if self.spacemouse_arm == "right":
            active_eef = right_eef
            active_gripper = right_gripper
            active_client = self.arx5_right_client
            inactive_eef = left_eef
            inactive_gripper = left_gripper
            inactive_client = self.arx5_left_client
        else:
            active_eef = left_eef
            active_gripper = left_gripper
            active_client = self.arx5_left_client
            inactive_eef = right_eef
            inactive_gripper = right_gripper
            inactive_client = self.arx5_right_client

        arm_switched = self.spacemouse_arm != self._last_spacemouse_arm
        needs_reseed = self.last_control_mode == TaskControlMode.POLICY or arm_switched or self._inactive_held_eef is None
        if needs_reseed:
            # Re-seed from the active arm's actual state to avoid a jump
            self.spacemouse_history_actions = {
                "action0_eef_xyz_wxyz": active_eef[np.newaxis, :],
                "action0_gripper_width": active_gripper[np.newaxis, :],
                "timestamps": np.array([time.monotonic()]),
            }
            self._inactive_held_eef = inactive_eef.copy()
            self._inactive_held_gripper = inactive_gripper.copy()
            if self.last_control_mode == TaskControlMode.POLICY:
                self.reset_policy_agent()
        self._last_spacemouse_arm = self.spacemouse_arm

        actions = self.spacemouse_agent.predict_actions(
            observations={
                "robot0_eef_xyz_wxyz": active_eef[np.newaxis, :],
                "robot0_gripper_width": active_gripper[np.newaxis, :],
            },
            history_actions=self.get_history_actions(
                length=self.spacemouse_agent.action_history_len
            ),
        )

        next_control_timestamp = (
            self.last_control_timestamp + 1.0 / self.spacemouse_agent.agent_update_freq_hz
        )
        wait_until(monotonic_time=next_control_timestamp)
        self.last_control_timestamp = next_control_timestamp

        dt = 1.0 / self.spacemouse_agent.agent_update_freq_hz
        active_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=actions["action0_eef_xyz_wxyz"][:, :],
            timestamps=np.array([dt + self.arx5_lookahead_time_s]),
            use_relative_timestamps=True,
        )
        active_client.schedule_joint_traj(
            joint_traj_pos=actions["action0_gripper_width"][:, :],
            timestamps=np.array([dt]),
            use_relative_timestamps=True,
        )
        # Inactive arm: hold a stable seeded pose (not the measured state) to prevent
        # FK noise from being fed back as the next IK target, which causes wrist drift.
        inactive_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=self._inactive_held_eef[np.newaxis, :],
            timestamps=np.array([dt + self.arx5_lookahead_time_s]),
            use_relative_timestamps=True,
        )
        inactive_client.schedule_joint_traj(
            joint_traj_pos=self._inactive_held_gripper[np.newaxis, :],
            timestamps=np.array([dt]),
            use_relative_timestamps=True,
        )

        self.spacemouse_history_actions = copy.deepcopy(actions)
        self.last_control_mode = TaskControlMode.SPACEMOUSE

    # ------------------------------------------------------------------
    # History actions (mirrors UmiARX5Task.get_history_actions)
    # ------------------------------------------------------------------

    def get_history_actions(self, length: int, action_type: str = "latest"):
        if len(self.policy_history_actions) == 0:
            history_actions = self.spacemouse_history_actions
        elif action_type == "latest":
            if (
                self.policy_history_actions["timestamps"][-1]
                > self.spacemouse_history_actions["timestamps"][-1]
            ):
                history_actions = self.policy_history_actions
            else:
                history_actions = self.spacemouse_history_actions
        elif action_type == "spacemouse":
            history_actions = self.spacemouse_history_actions
        elif action_type == "policy":
            history_actions = self.policy_history_actions
        else:
            raise ValueError(f"Invalid action type: {action_type}")

        processed: dict[str, npt.NDArray[np.float64]] = {}
        for key, value in history_actions.items():
            if value.shape[0] > length:
                processed[key] = value[-length:]
            elif value.shape[0] < length:
                processed[key] = np.concatenate(
                    [np.repeat(value[:1, ...], length - value.shape[0], axis=0), value]
                )
            else:
                processed[key] = value
        return processed

    # ------------------------------------------------------------------
    # Display / disconnect
    # ------------------------------------------------------------------

    def display_robot_state(self):
        left_eef = self.arx5_left_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        left_gripper = self.arx5_left_client.get_joint_pos(timestamp=time.monotonic())
        right_eef = self.arx5_right_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        right_gripper = self.arx5_right_client.get_joint_pos(timestamp=time.monotonic())
        print(f"Left  EEF: {left_eef},  gripper: {left_gripper}")
        print(f"Right EEF: {right_eef}, gripper: {right_gripper}")
        print(f"SpaceMouse arm: {self.spacemouse_arm}")

    def disconnect(self):
        pass


def run_iphumi_arx5_bimanual_task():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    task_name = sys.argv[1] if len(sys.argv) > 1 else None
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    config_name = f"iphumi_arx5_bimanual_{task_name}" if task_name else "iphumi_arx5_bimanual"
    with hydra.initialize(config_path="../configs/tasks"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    print(cfg)
    task: iPhumiARX5BimanualTask = hydra.utils.instantiate(cfg)
    try:
        task.run()
    except KeyboardInterrupt:
        pass
    finally:
        os.system("stty sane")


if __name__ == "__main__":
    run_iphumi_arx5_bimanual_task()
