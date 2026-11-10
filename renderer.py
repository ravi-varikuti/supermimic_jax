import mujoco
import numpy as np
import viser
from threading import RLock


class ViserRenderer:
    """Display a selected world or a grid of all worlds."""

    def __init__(self, model, num_envs=1, port=8080):
        self.model = model
        self.num_envs = num_envs
        self.data = mujoco.MjData(model)
        self._lock = RLock()
        self._qpos = None
        self._reference_qpos = None
        self.server = viser.ViserServer(host="127.0.0.1", port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/floor", plane="xy", infinite_grid=True)
        self.env_selector = self.server.gui.add_dropdown(
            "Environment", options=[str(i) for i in range(num_envs)], initial_value="0",
        )
        self.show_all = self.server.gui.add_checkbox("Show all environments", False)
        self.show_reference = self.server.gui.add_checkbox("Show reference (cyan)", True)
        self.robots = {}
        self.references = {}

        @self.env_selector.on_update
        @self.show_all.on_update
        @self.show_reference.on_update
        def change_view(_):
            self._draw()
            for client in self.server.get_clients().values():
                self._set_camera(client)

        self.server.on_client_connect(self._set_camera)

    def _add_robot(self, index, reference=False):
        path = f"/reference_{index}" if reference else f"/env_{index}"
        root = self.server.scene.add_frame(path, show_axes=False)
        meshes = []
        model = self.model

        for geom in range(model.ngeom):
            # Unitree's visual meshes are separate from collision geometry.
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 1:
                continue
            mesh = model.geom_dataid[geom]
            v, f = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
            vertices = model.mesh_vert[v:v + model.mesh_vertnum[mesh]]
            faces = model.mesh_face[f:f + model.mesh_facenum[mesh]]
            color = tuple((255 * model.geom_rgba[geom, :3]).astype(int))
            handle = self.server.scene.add_mesh_simple(
                f"{path}/geom_{geom}", vertices, faces,
                color=(0, 180, 255) if reference else color,
                opacity=0.5 if reference else 1.0,
            )
            meshes.append((geom, handle))
        collection = self.references if reference else self.robots
        collection[index] = (root, meshes)

    def _set_camera(self, client):
        columns = int(np.ceil(np.sqrt(self.num_envs)))
        rows = int(np.ceil(self.num_envs / columns))
        center = np.array([0.0, 0.0, 0.8])
        distance = 2.0
        if self.show_all.value:
            center[:2] = 1.25 * np.array([columns - 1, rows - 1])
            distance = max(2.0, 1.8 * max(columns, rows))
        client.camera.position = center + np.array([distance, -distance, 0.7 * distance])
        client.camera.look_at = center
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def update(self, qpos, reference_qpos=None):
        with self._lock:
            self._qpos = np.asarray(qpos).copy()
            self._reference_qpos = None if reference_qpos is None else np.asarray(reference_qpos).copy()
            self._draw()

    def _update_pose(self, meshes, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        for geom, handle in meshes:
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[geom])
            handle.position = self.data.geom_xpos[geom].copy()
            handle.wxyz = quat

    def _draw(self):
        with self._lock:
            if self._qpos is None:
                return
            indices = range(self.num_envs) if self.show_all.value else [int(self.env_selector.value)]
            columns = int(np.ceil(np.sqrt(self.num_envs)))
            with self.server.atomic():
                groups = ((False, self.robots, self._qpos),
                          (True, self.references, self._reference_qpos))
                for reference, collection, poses in groups:
                    active = indices if poses is not None and (not reference or self.show_reference.value) else []
                    for index in active:
                        if index not in collection:
                            self._add_robot(index, reference=reference)
                    for index, (root, meshes) in collection.items():
                        root.visible = index in active
                        if not root.visible:
                            continue
                        # Both robots share the same display offset.
                        offset = np.array([2.5 * (index % columns), 2.5 * (index // columns), 0]) if self.show_all.value else np.zeros(3)
                        root.position = offset
                        self._update_pose(meshes, poses[index])

    def close(self):
        self.server.stop()
