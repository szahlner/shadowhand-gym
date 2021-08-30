import os

import numpy as np
import time
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple


class PyBullet:
    def __init__(
        self,
        render: bool = False,
        n_substeps: int = 10,
        position_gain: float = 0.02,
        background_color: Optional[List[int]] = None,
    ) -> None:
        """Convenient class to use PyBullet physics engine.

        Args:
            render (bool, optional): Enable rendering. Defaults to False.
            n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
            position_gain (float, optional): Positional gain to control the joints. Defaults to 0.02.
            background_color (List[int], optional): Simulator background color [R, G, B]. Defaults to [210, 200, 190].
        """
        if background_color is None:
            background_color = [210, 200, 190]
        assert len(background_color) == 3, "Background color must be a list of 3 values"

        self.render_enabled = render
        self.background_color = [val / 255 for val in background_color]

        if render:
            options = (
                "--background_color_red={} "
                "--background_color_green={} "
                "--background_color_blue={}".format(*self.background_color)
            )
            self.physics_client = bc.BulletClient(
                connection_mode=p.GUI, options=options
            )
            self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        else:
            self.physics_client = bc.BulletClient(connection_mode=p.DIRECT)

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 240  # 500

        self.position_gain = position_gain

        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)

        self._bodies_idx = {}

        # MP4 video logging
        self._logger_id = None

    @property
    def dt(self) -> float:
        """Timestep.

        Returns:
            float: Timestep.
        """
        return self.timestep * self.n_substeps

    def step(self) -> None:
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        p.disconnect()

    def render(
        self,
        mode: str = "human",
        width: int = 384,
        height: int = 384,
        target_position: Optional[List[float]] = None,
        distance: float = 0.5,
        yaw: float = 45,
        pitch: float = -40,
        roll: float = 0,
    ) -> None:
        """Render.

        If mode is 'human', make the rendering real-time.
        If mode is 'rgb_array', return rgb_array of the scene.
        If mode is 'mp4', return mp4 video.
        All other arguments are unused.

        Args:
            mode (str, optional): 'human', 'rgb_array' or 'mp4'. If human, just sleep a few time to make the rendering
                real-time, else, return an RGB array. Defaults to 'human'.
            width (int, optional): Image width. Defaults to 920.
            height (int, optional): Image height. Defaults to 720.
            target_position (List[float], optional): Camera targeting this position in cartesian coordinates [x, y, z].
                Defaults to [0.25, 0.0, 0.0].
            distance (float, optional): Distance of the camera. Defaults to 2 m.
            yaw (float, optional): Yaw of the camera. Defaults to 45 degree.
            pitch (float, optional): Pitch of the camera. Defaults to -15 degree.
            roll (float, optional): Roll of the camera. Defaults to 0 degree.
        """
        if target_position is None:
            target_position = [0.25, 0.0, 0.0]
        assert len(target_position) == 3, "Target position must be a list of 3 values"

        if mode == "human":
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING
            )
            time.sleep(self.dt)  # wait to seems like real speed
        elif mode == "rgb_array":
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, px, depth, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
            )

            # configure background color
            bg = [val * 255 for val in self.background_color] + [255.0]
            for ix in range(len(px)):
                for iy in range(len(px[ix])):
                    if depth[ix][iy] > 0.99:
                        px[ix][iy][:] = bg

            rgb_array = np.reshape(np.array(px), (height, width, 4))
            rgb_array = rgb_array[:, :, :3]

            return rgb_array
        elif mode == "mp4":
            self._logger_id = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName="filename.mp4"
            )
        else:
            assert False, "Mode '{}' not implemented".format(mode)

    def get_base_position(self, body: str) -> Tuple[float, float, float]:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (float, float, float): The cartesian position (x, y, z).
        """
        return self.physics_client.getBasePositionAndOrientation(
            self._bodies_idx[body]
        )[0]

    def get_base_orientation(self, body: str) -> Tuple[float, float, float]:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (float, float, float): The orientation as quaternion (x, y, z, w).
        """
        return self.physics_client.getBasePositionAndOrientation(
            self._bodies_idx[body]
        )[1]

    def get_base_rotation(self, body: str) -> Tuple[float, float, float]:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (float, float, float): The rotation in euler notation (yaw, pitch, roll).
        """
        return self.physics_client.getEulerFromQuaternion(
            self.get_base_orientation(body)
        )

    def get_base_velocity(self, body: str) -> Tuple[float, float, float]:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (float, float, float): The cartesian velocity (vx, vy, vz).
        """
        return self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]

    def get_base_angular_velocity(self, body: str) -> Tuple[float, float, float]:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (float, float, float): The angular velocity (wx, wy, wz).
        """
        return self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]

    def get_link_position(self, body: str, link: int) -> Tuple[float, float, float]:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (float, float, float): The cartesian position (x, y, z).
        """
        return self.physics_client.getLinkState(self._bodies_idx[body], link)[0]

    def get_link_orientation(
        self, body: str, link: int
    ) -> Tuple[float, float, float, float]:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (float, float, float, float): The orientation as quaternion (x, y, z, w).
        """
        return self.physics_client.getLinkState(self._bodies_idx[body], link)[1]

    def get_link_velocity(self, body: str, link: int) -> Tuple[float, float, float]:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (float, float, float): The cartesian velocity (vx, vy, vz).
        """
        return self.physics_client.getLinkState(
            self._bodies_idx[body], link, computeLinkVelocity=True
        )[6]

    def get_link_angular_velocity(
        self, body: str, link: int
    ) -> Tuple[float, float, float]:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (float, float, float): The angular velocity (wx, wy, wz).
        """
        return self.physics_client.getLinkState(
            self._bodies_idx[body], link, computeLinkVelocity=True
        )[7]

    def get_joint_position(self, body: str, joint: int) -> float:
        """Get the position/angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.

        Returns:
            float: The position/angle of the joint.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.

        Returns:
            float: The velocity of the joint.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[1]

    def set_base_pose(
        self, body: str, target_position: List[float], target_orientation: List[float]
    ) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            target_position (List[float]): The target cartesian position [x, y, z].
            target_orientation (List[float]): The target orientation as quaternion [x, y, z, w].
        """
        assert (
            len(target_position) == 3
        ), "Target position must be of length 3: [x, y, z]"
        assert (
            len(target_orientation) == 4
        ), "Target orientation must be of length 4: [x, y,z, w]"

        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body],
            posObj=target_position,
            ornObj=target_orientation,
        )

    def set_joint_positions(
        self, body: str, joint_indices: List[int], positions: List[float]
    ) -> None:
        """Set the positions/angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joint_indices (List[int]): List of joint indices.
            positions (List[float]): List of target angles.
        """
        for joint, position in zip(joint_indices, positions):
            self.set_joint_position(body=body, joint_idx=joint, position=position)

    def set_joint_position(self, body: str, joint_idx: int, position: float) -> None:
        """Set the position/angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint_idx (int): Joint index in the body.
            position (float): Target angle.
        """
        self.physics_client.resetJointState(
            bodyUniqueId=self._bodies_idx[body],
            jointIndex=joint_idx,
            targetValue=position,
        )

    def control_joints(
        self,
        body: str,
        joint_indices: List[int],
        target_positions: List[float],
        target_forces: List[float],
    ) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joint_indices (List[int]): List of joint indices.
            target_positions (List[float]): List of target positions/angles.
            target_forces (List[float]): Forces to apply.
        """
        position_gains = [self.position_gain] * len(joint_indices)

        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joint_indices,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=target_forces,
            positionGains=position_gains,
        )

    def place_visualizer(
        self, target_position: List[float], distance: float, yaw: float, pitch: float
    ) -> None:
        """Orient the camera used for rendering.

        Args:
            target_position (List[float]): Target cartesian position [x, y, z].
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self) -> None:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_RENDERING, 0
        )
        yield
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_RENDERING, 1
        )

    def loadURDF(self, body_name: str, **kwargs: dict) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        kwargs["flags"] = p.URDF_USE_SELF_COLLISION
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def get_num_joints(self, body_name: str) -> int:
        """Get total number of joints in the robot.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.

        Returns:
            int: Total number of joints in the robot.
        """
        return self.physics_client.getNumJoints(self._bodies_idx[body_name])

    def get_joint_lower_limit(self, body: str, joint: int) -> float:
        """Get the lower limit of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.

        Returns:
            float: The lower limit of the joint.
        """
        return self.physics_client.getJointInfo(self._bodies_idx[body], joint)[8]

    def get_joint_upper_limit(self, body: str, joint: int) -> float:
        """Get the upper limit of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.

        Returns:
            float: The upper limit of the joint.
        """
        return self.physics_client.getJointInfo(self._bodies_idx[body], joint)[9]

    def get_joint_name(self, body: str, joint: int) -> str:
        """Get the name of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.

        Returns:
            str: The name of the joint.
        """
        return self.physics_client.getJointInfo(self._bodies_idx[body], joint)[1]

    def create_box(
        self,
        body_name: str,
        half_extents: List[float],
        mass: float,
        position: List[float],
        rgba_color: List[float],
        specular_color: Optional[List[float]] = None,
        ghost: bool = False,
        friction: Optional[float] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (List[float]): Half size of the box in metres [x, y, z].
            mass (float): The mass in kg.
            position (List[float]): The cartesian position of the box [x, y, z].
            rgba_color (List[float]): RGBA color [R, G, B, A].
            specular_color (List[float], optional): RGB specular color. Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the box can collide or not. Defaults to False.
            friction (float, optional): The friction. If None, keep the PyBullet default value. Defaults to None.
        """
        if specular_color is None:
            specular_color = [0.0, 0.0, 0.0]

        assert len(half_extents) == 3, "Half size must be a list of length 3: [x, y, z]"
        assert len(position) == 3, "Position must be of length 3: [x, y, z]"
        assert len(rgba_color) == 4, "RGBA color must be of length 4: [R, G, B, A]"
        assert (
            len(specular_color) == 3
        ), "RGB specular color must be of length 3: [R, G, B]"

        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}

        return self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: List[float],
        rgba_color: List[float],
        specular_color: Optional[List[float]] = None,
        ghost: bool = False,
        friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in metres.
            height (float): The height in metres.
            mass (float): The mass in kg.
            position (List[float]): The cartesian position of the box [x, y, z].
            rgba_color (List[float]): RGBA color [R, G, B, A].
            specular_color (List[float], optional): RGB specular color. Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the box can collide or not. Defaults to False.
            friction (float, optional): The friction. If None, keep the PyBullet default value. Defaults to None.
        """
        if specular_color is None:
            specular_color = [0.0, 0.0, 0.0]

        assert len(position) == 3, "Position must be of length 3: [x, y, z]"
        assert len(rgba_color) == 4, "RGBA color must be of length 4: [R, G, B, A]"
        assert (
            len(specular_color) == 3
        ), "RGB specular color must be of length 3: [R, G, B]"

        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}

        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: List[float],
        rgba_color: List[float],
        specular_color: Optional[List[float]] = None,
        ghost: bool = False,
        friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in metres.
            mass (float): The mass in kg.
            position (float, float ,float): The position of the box.
            rgba_color (float, float, float, float): RGBA color.
            specular_color (float, float, float, float): RGB specular color. Defaults to (0, 0, 0).
            ghost (bool, optional): Whether the box can collide or not. Defaults to False.
            friction (float, optional): The friction. If None, keep the PyBullet default value. Defaults to None.
        """
        if specular_color is None:
            specular_color = [0.0, 0.0, 0.0]

        assert len(position) == 3, "Position must be of length 3: [x, y, z]"
        assert len(rgba_color) == 4, "RGBA color must be of length 4: [R, G, B, A]"
        assert (
            len(specular_color) == 3
        ), "RGB specular color must be of length 3: [R, G, B]"

        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}

        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: Optional[List[float]] = None,
        ghost: float = False,
        friction: Optional[float] = None,
        visual_kwargs: Optional[Dict] = None,
        collision_kwargs: Optional[Dict] = None,
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (List[float]): The cartesian position of the geometry. Defaults to [0.0, 0.0, 0.0].
            ghost (bool, optional): Whether the box can collide or not. Defaults to False.
            friction (float, optional): The friction coefficient.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        if position is None:
            position = [0.0, 0.0, 0.0]

        if visual_kwargs is None:
            visual_kwargs = {}

        if collision_kwargs is None:
            collision_kwargs = {}

        base_visual_shape_index = self.physics_client.createVisualShape(
            geom_type, **visual_kwargs
        )

        if not ghost:
            base_collision_shape_index = self.physics_client.createCollisionShape(
                geom_type, **collision_kwargs
            )
        else:
            base_collision_shape_index = -1

        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=base_visual_shape_index,
            baseCollisionShapeIndex=base_collision_shape_index,
            baseMass=mass,
            basePosition=position,
        )

        if friction is not None:
            self.physics_client.changeDynamics(
                bodyUniqueId=self._bodies_idx[body_name],
                linkIndex=-1,
                lateralFriction=friction,
            )

    def create_plane(self, z_offset: float) -> None:
        """Create a plane.

        Actually it is a thin box.

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=[3.0, 3.0, 0.01],
            mass=0,
            position=[0.0, 0.0, z_offset - 0.01],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.95, 0.95, 0.95, 1],
            friction=0.1,
        )

    def create_table(
        self, length: float, width: float, height: float, x_offset: float = 0.0
    ) -> None:
        """Create a fixed table.

        Top is z=0, centered in y.

        Args:
            length (float): Table length.
            width (float): Table width.
            height (float): Table height.
            x_offset (float, optional): X offset.
        """
        self.create_box(
            body_name="table",
            half_extents=[length / 2, width / 2, height / 2],
            mass=0,
            position=[x_offset, 0.0, -height / 2],
            rgba_color=[0.95, 0.95, 0.95, 1],
            friction=0.1,
        )

    def set_friction(self, body: str, link: int, friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=friction,
        )

    def create_object(
        self,
        body_name: str,
        object_path: str,
        texture_path: str,
        mass: float = 0.0,
        position: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        rgba_color: Optional[List[float]] = None,
        friction: Optional[float] = None,
        mesh_scale: Optional[List[float]] = None,
    ) -> None:
        """Create an object.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            object_path (str): Path to the object file.
            texture_path (str): Path to the texture.
            mass (float, optional): The mass in kg. Defaults to 0.0.
            position (List[float]): The cartesian position of the geometry. Defaults to [0.0, 0.0, 0.0].
            orientation (List[float]): The orientation of the geometry in quaternions. Defaults to [0.0, 0.0, 0.0, 0.1].
            rgba_color (float, float, float, float): RGBA color.
            friction (float): Lateral friction.
            mesh_scale (List[float]): Scale of the mesh in x, y and z direction. Defaults to [1.0, 1.0, 1.0].
        """
        if position is None:
            position = [0.0] * 3

        if orientation is None:
            orientation = [0.0, 0.0, 0.0, 1.0]

        if mesh_scale is None:
            mesh_scale = [1.0] * 3

        assert len(position) == 3, "Position ('position') must be of length 3"
        assert len(orientation) == 4, "Orientation ('orientation') must be of length 4"
        assert len(mesh_scale) == 3, "Mesh scale ('mesh_scale') must be of length 3"
        assert os.path.exists(object_path), "Object path ('object_path') does not exist"
        assert os.path.exists(
            texture_path
        ), "Texture path ('texture_path') does not exist"

        visual_shape_id = self.physics_client.createVisualShape(
            fileName=object_path,
            shapeType=p.GEOM_MESH,
            rgbaColor=rgba_color,
            meshScale=mesh_scale,
        )

        collision_shape_id = self.physics_client.createCollisionShape(
            fileName=object_path, shapeType=p.GEOM_MESH, meshScale=mesh_scale
        )

        texture_id = self.physics_client.loadTexture(texture_path)

        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseMass=mass,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=position,
            baseOrientation=orientation,
        )

        self.physics_client.changeVisualShape(
            objectUniqueId=self._bodies_idx[body_name],
            linkIndex=-1,
            textureUniqueId=texture_id,
        )

        if friction is not None:
            self.physics_client.changeDynamics(
                bodyUniqueId=self._bodies_idx[body_name],
                linkIndex=-1,
                lateralFriction=friction,
            )
