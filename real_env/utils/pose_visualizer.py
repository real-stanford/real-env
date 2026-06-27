"""Live Open3D visualizer for bimanual arm EEF poses.

World origin = left arm base frame.
  - Large coordinate frame at origin = left arm base
  - Medium coordinate frame = right arm base (static, from tx_left_right_base)
  - Coordinate frame + blue sphere  = left arm EEF (live, size 0.10)
  - Coordinate frame + red sphere   = right arm EEF (live, size 0.10)
  - Blue wireframe axes (size 0.10) = left EEF start pose (ghost, frozen on episode start)
  - Red  wireframe axes (size 0.10) = right EEF start pose (ghost, frozen on episode start)
  - Blue wireframe axes (size 0.13) = left EEF reconstructed via cross-arm obs (halo)
  - Red  wireframe axes (size 0.13) = right EEF reconstructed via cross-arm obs (halo)

Ghost frames:   T_ghost = T_eef_cur @ inv(T_wrt_start)  → must stay frozen.
Halo frames:    T_halo_left  = T_right_eef_world @ T_wrt_right  → must track left EEF.
                T_halo_right = T_left_eef_world  @ T_wrt_left   → must track right EEF.
Both appear only during policy execution (when obs are available).
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


class PoseVisualizer:
    def __init__(self, tx_left_right_base: np.ndarray, update_hz: float = 30.0):
        """
        Args:
            tx_left_right_base: 4x4 SE3 matrix transforming from right arm base frame
                                 to left arm base frame.
            update_hz: target render rate for the Open3D window.
        """
        self.tx_left_right_base = np.array(tx_left_right_base, dtype=np.float64)
        self.update_hz = update_hz

        self._lock = threading.Lock()
        self._left_xyz_wxyz: np.ndarray | None = None
        self._right_xyz_wxyz: np.ndarray | None = None
        self._new_data: bool = False

        # Ghost (start pose) transforms
        self._ghost_left_T: np.ndarray | None = None
        self._ghost_right_T: np.ndarray | None = None
        self._new_ghost_data: bool = False

        # Halo (cross-arm reconstructed) transforms
        self._halo_left_T: np.ndarray | None = None
        self._halo_right_T: np.ndarray | None = None
        self._new_halo_data: bool = False

        self._thread: threading.Thread | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        left_xyz_wxyz: np.ndarray,
        right_xyz_wxyz: np.ndarray,
        left_wrt_start_xyz_wxyz: np.ndarray | None = None,
        right_wrt_start_xyz_wxyz: np.ndarray | None = None,
        left_wrt_right_xyz_wxyz: np.ndarray | None = None,
        right_wrt_left_xyz_wxyz: np.ndarray | None = None,
    ) -> None:
        """Push latest EEF poses. Thread-safe; call from the main control loop."""
        # All computation outside the lock
        left_T = self._xyz_wxyz_to_matrix(left_xyz_wxyz)
        right_T_in_right = self._xyz_wxyz_to_matrix(right_xyz_wxyz)
        right_T_in_left = self.tx_left_right_base @ right_T_in_right

        ghost_left_T = ghost_right_T = None
        if left_wrt_start_xyz_wxyz is not None and right_wrt_start_xyz_wxyz is not None:
            # T_ghost = T_eef_cur @ inv(T_wrt_start)  →  should equal T_start
            ghost_left_T = left_T @ np.linalg.inv(
                self._xyz_wxyz_to_matrix(left_wrt_start_xyz_wxyz)
            )
            ghost_right_in_right = right_T_in_right @ np.linalg.inv(
                self._xyz_wxyz_to_matrix(right_wrt_start_xyz_wxyz)
            )
            ghost_right_T = self.tx_left_right_base @ ghost_right_in_right

        halo_left_T = halo_right_T = None
        if left_wrt_right_xyz_wxyz is not None and right_wrt_left_xyz_wxyz is not None:
            # Reconstruct left EEF from right EEF + cross-arm relative obs
            # T_halo_left = T_right_eef_in_left @ T_wrt_right  →  should equal T_left
            halo_left_T = right_T_in_left @ self._xyz_wxyz_to_matrix(left_wrt_right_xyz_wxyz)

            # Reconstruct right EEF from left EEF + cross-arm relative obs
            # T_halo_right = tx @ (inv(tx) @ T_left @ T_wrt_left)  →  should equal T_right_in_left
            left_T_in_right = np.linalg.inv(self.tx_left_right_base) @ left_T
            halo_right_in_right = left_T_in_right @ self._xyz_wxyz_to_matrix(right_wrt_left_xyz_wxyz)
            halo_right_T = self.tx_left_right_base @ halo_right_in_right

        with self._lock:
            self._left_xyz_wxyz = left_xyz_wxyz.copy()
            self._right_xyz_wxyz = right_xyz_wxyz.copy()
            self._new_data = True
            if ghost_left_T is not None:
                self._ghost_left_T = ghost_left_T
                self._ghost_right_T = ghost_right_T
                self._new_ghost_data = True
            if halo_left_T is not None:
                self._halo_left_T = halo_left_T
                self._halo_right_T = halo_right_T
                self._new_halo_data = True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            print("[PoseVis] Already running.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PoseVisualizer"
        )
        self._thread.start()
        print(
            "[PoseVis] Visualization window opened.\n"
            "          Large frame @ origin          = left arm base\n"
            "          Medium frame                  = right arm base\n"
            "          Blue sphere + frame (0.10)    = left EEF (live)\n"
            "          Red  sphere + frame (0.10)    = right EEF (live)\n"
            "          Blue wireframe      (0.10)    = left EEF start pose (ghost — must stay frozen)\n"
            "          Red  wireframe      (0.10)    = right EEF start pose (ghost — must stay frozen)\n"
            "          Blue wireframe halo (0.13)    = left EEF reconstructed via cross-arm obs\n"
            "          Red  wireframe halo (0.13)    = right EEF reconstructed via cross-arm obs\n"
            "          Halos appear on episode start and must track the live EEF frames.\n"
            "          Press 'v' again to close."
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[PoseVis] Visualization stopped.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _xyz_wxyz_to_matrix(xyz_wxyz: np.ndarray) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 3] = xyz_wxyz[:3]
        w, x, y, z = xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6]
        mat[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        return mat

    @staticmethod
    def _apply_delta(geom, new_T: np.ndarray, prev_T: np.ndarray) -> None:
        geom.transform(new_T @ np.linalg.inv(prev_T))

    @staticmethod
    def _make_wireframe_axes(size: float, color: list):
        """RGB wireframe axes as a LineSet (X=red, Y=green, Z=blue), tinted by color."""
        import open3d as o3d
        pts = np.array([
            [0, 0, 0], [size, 0, 0],
            [0, 0, 0], [0, size, 0],
            [0, 0, 0], [0, 0, size],
        ], dtype=np.float64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
        ls.colors = o3d.utility.Vector3dVector([color, color, color])
        return ls

    # ------------------------------------------------------------------
    # Render loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            "Bimanual EEF Poses  |  world = left arm base frame",
            width=1000,
            height=750,
        )

        opt = vis.get_render_option()
        opt.background_color = np.array([0.15, 0.15, 0.15])
        opt.point_size = 3.0

        # ---- Static geometry ------------------------------------------------
        left_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(left_base_frame)

        right_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        right_base_frame.transform(self.tx_left_right_base)
        vis.add_geometry(right_base_frame)

        # ---- Live EEF geometry (size 0.10) ----------------------------------
        left_eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        left_eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        left_eef_sphere.paint_uniform_color([0.2, 0.4, 1.0])
        left_eef_sphere.compute_vertex_normals()
        vis.add_geometry(left_eef_frame)
        vis.add_geometry(left_eef_sphere)

        right_eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        right_eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        right_eef_sphere.paint_uniform_color([1.0, 0.3, 0.2])
        right_eef_sphere.compute_vertex_normals()
        vis.add_geometry(right_eef_frame)
        vis.add_geometry(right_eef_sphere)

        # ---- Ghost axes: start pose (size 0.10, same as live) ---------------
        ghost_left_axes = self._make_wireframe_axes(size=0.10, color=[0.4, 0.7, 1.0])
        ghost_right_axes = self._make_wireframe_axes(size=0.10, color=[1.0, 0.6, 0.4])
        vis.add_geometry(ghost_left_axes)
        vis.add_geometry(ghost_right_axes)

        # ---- Halo axes: cross-arm reconstructed (size 0.13, larger) --------
        halo_left_axes = self._make_wireframe_axes(size=0.13, color=[0.2, 0.4, 1.0])
        halo_right_axes = self._make_wireframe_axes(size=0.13, color=[1.0, 0.3, 0.2])
        vis.add_geometry(halo_left_axes)
        vis.add_geometry(halo_right_axes)

        # ---- Default camera view --------------------------------------------
        view_ctl = vis.get_view_control()
        view_ctl.set_front([-1.0, 0.0, 0.0])
        view_ctl.set_up([0.0, 0.0, 1.0])
        view_ctl.set_lookat([0.0, -0.2, 0.15])
        view_ctl.set_zoom(0.45)

        # ---- Transform tracking ---------------------------------------------
        left_T_cur = np.eye(4)
        right_T_cur = np.eye(4)
        left_sph_T_cur = np.eye(4)
        right_sph_T_cur = np.eye(4)
        ghost_left_T_cur = np.eye(4)
        ghost_right_T_cur = np.eye(4)
        halo_left_T_cur = np.eye(4)
        halo_right_T_cur = np.eye(4)

        dt = 1.0 / self.update_hz

        while self._running:
            t0 = time.monotonic()

            with self._lock:
                new_data = self._new_data
                new_ghost = self._new_ghost_data
                new_halo = self._new_halo_data
                if new_data:
                    left_xyz_wxyz = self._left_xyz_wxyz.copy()
                    right_xyz_wxyz = self._right_xyz_wxyz.copy()
                    self._new_data = False
                if new_ghost:
                    ghost_left_T = self._ghost_left_T.copy()
                    ghost_right_T = self._ghost_right_T.copy()
                    self._new_ghost_data = False
                if new_halo:
                    halo_left_T = self._halo_left_T.copy()
                    halo_right_T = self._halo_right_T.copy()
                    self._new_halo_data = False

            if new_data:
                left_T = self._xyz_wxyz_to_matrix(left_xyz_wxyz)
                right_T = self.tx_left_right_base @ self._xyz_wxyz_to_matrix(right_xyz_wxyz)

                self._apply_delta(left_eef_frame, left_T, left_T_cur)
                vis.update_geometry(left_eef_frame)
                left_T_cur = left_T

                self._apply_delta(right_eef_frame, right_T, right_T_cur)
                vis.update_geometry(right_eef_frame)
                right_T_cur = right_T

                left_sph_T = np.eye(4)
                left_sph_T[:3, 3] = left_T[:3, 3]
                self._apply_delta(left_eef_sphere, left_sph_T, left_sph_T_cur)
                vis.update_geometry(left_eef_sphere)
                left_sph_T_cur = left_sph_T

                right_sph_T = np.eye(4)
                right_sph_T[:3, 3] = right_T[:3, 3]
                self._apply_delta(right_eef_sphere, right_sph_T, right_sph_T_cur)
                vis.update_geometry(right_eef_sphere)
                right_sph_T_cur = right_sph_T

            if new_ghost:
                self._apply_delta(ghost_left_axes, ghost_left_T, ghost_left_T_cur)
                vis.update_geometry(ghost_left_axes)
                ghost_left_T_cur = ghost_left_T

                self._apply_delta(ghost_right_axes, ghost_right_T, ghost_right_T_cur)
                vis.update_geometry(ghost_right_axes)
                ghost_right_T_cur = ghost_right_T

            if new_halo:
                self._apply_delta(halo_left_axes, halo_left_T, halo_left_T_cur)
                vis.update_geometry(halo_left_axes)
                halo_left_T_cur = halo_left_T

                self._apply_delta(halo_right_axes, halo_right_T, halo_right_T_cur)
                vis.update_geometry(halo_right_axes)
                halo_right_T_cur = halo_right_T

            try:
                vis.poll_events()
                vis.update_renderer()
            except Exception:
                break

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))

        try:
            vis.destroy_window()
        except Exception:
            pass
