import pyvista as pv
import numpy as np
import os
import util
from pathlib import Path
from scipy.spatial.transform import Rotation

import logging

# ########################################
#           SETUP
# ########################################

logger = logging.getLogger(__name__)
log_file = Path(__file__).with_suffix(".log")
logging.basicConfig(filename=log_file, level=logging.INFO)

OFFSCREEN = False
plotter = util.setup_plotter(
    offscreen=OFFSCREEN, skybox=util.SkyBox.grey, resolution=(1024, 720)
)
plotter.disable_shadows()

MESH_ARGS = {
    "smooth_shading": True,
    "pbr": True,
    "metallic": 0.3,
    "roughness": 0.85,
}


def normalize(arr):
    return arr / np.linalg.norm(arr)


def proj_tangential(arr1, arr2):
    return arr1 - np.dot(arr1, arr2) * arr2


def proj_parallel(arr1, arr2):
    return np.dot(arr1, arr2) * arr2


point1 = normalize(np.array([0.4, 0, 1]))
point2 = normalize(np.array([-0.4, 0, 1]))

# The gradient force on the spins
force1 = normalize(np.cross([3, 1, 2], point1))
force2 = 0.8 * normalize(np.cross([-2.5, 2, 2], point2))

# The tangents at the spins
tangent_1 = normalize(proj_tangential(point2 - point1, point1))
tangent_2 = normalize(proj_tangential(point2 - point1, point2))

# Projected forces
force_1_par = proj_parallel(force1, tangent_1)
force_1_orth = proj_tangential(force1, tangent_1)
force_2_par = proj_parallel(force2, tangent_2)
force_2_orth = proj_tangential(force2, tangent_2)

angle = np.arccos(np.dot(point1, point2))
axis = normalize(np.cross(point1, point2))

rotation_matrix = Rotation.from_rotvec(angle * axis).as_matrix()

# force 1 in the frame of spin2
force_1_rotated = rotation_matrix.dot(force1)

# force 2 in the frame of spin1
force_2_rotated = rotation_matrix.transpose().dot(force2)

g_S_1 = 0.5 * (force1 + force_2_rotated)
g_A_1 = 0.5 * (force1 - force_2_rotated)
g_orth_1 = proj_tangential(g_S_1, tangent_1)
g_climb_1 = -proj_parallel(g_S_1, tangent_1)
g_rot_1 = proj_tangential(g_A_1, tangent_1)


# symmetrised gradient force in frame of spin1
g_S_2 = 0.5 * (force_1_rotated + force2)
g_A_2 = 0.5 * (force_1_rotated - force2)
g_orth_2 = proj_tangential(g_S_2, tangent_2)
g_climb_2 = -proj_parallel(g_S_2, tangent_2)
g_rot_2 = -proj_tangential(g_A_2, tangent_2)


force_1_par_rotated = rotation_matrix.dot(force_1_par)
force_2_par_rotated = rotation_matrix.transpose().dot(force_2_par)

center = np.array([0, 0, 0])


# ########################################
#           MESHES
# ########################################

# this arc is at the tip of the arrows, it marks the tangent
arc = pv.CircularArc(point1, point2, center).tube(radius=0.005)
plotter.add_mesh(arc, color="black", **MESH_ARGS)

# this arc is further down, it marks the angle
arc2 = pv.CircularArc(point1 / 3, point2 / 3, center).tube(radius=0.005)
plotter.add_mesh(arc2, color="black", **MESH_ARGS)

# Arrows to mark the directions of the spin
arrow_dict = dict(
    scale=1, tip_length=0.25 / 2, tip_radius=0.1 / 2, shaft_radius=0.05 / 2
)
plotter.add_mesh(pv.Arrow(center, point1, **arrow_dict), color="black", **MESH_ARGS)
plotter.add_mesh(pv.Arrow(center, point2, **arrow_dict), color="lightgray", **MESH_ARGS)

# Tangent planes
plane = pv.Plane(point1, point1, i_size=0.7, j_size=0.7)
plotter.add_mesh(plane, opacity=0.4, **MESH_ARGS)

plane = pv.Plane(point2, point2, i_size=0.7, j_size=0.7)
plotter.add_mesh(plane, opacity=0.4, **MESH_ARGS)


# Force arrows
TIP_LENGTH = 0.05
TIP_RADIUS = 0.020 / 1.5
SHAFT_RADIUS = 0.015 / 2


def arrow_from_force(
    center,
    force,
    tip_length=TIP_LENGTH,
    tip_radius=TIP_RADIUS,
    shaft_radius=SHAFT_RADIUS,
):
    scale = 0.4 * np.linalg.norm(force)
    arrow_dict = dict(
        scale=scale,
        tip_length=tip_length / scale,
        tip_radius=tip_radius / scale,
        shaft_radius=shaft_radius / scale,
    )
    return pv.Arrow(center, force, **arrow_dict)


###### Frame of spin 1 ########

# gradient force

plotter.add_mesh(arrow_from_force(point1, force1), color="black", **MESH_ARGS)
plotter.add_mesh(arrow_from_force(point1, force_2_rotated), color="grey", **MESH_ARGS)
plotter.add_mesh(
    arrow_from_force(point1, g_orth_1, shaft_radius=SHAFT_RADIUS * 1.05),
    color="red",
    **MESH_ARGS,
)
plotter.add_mesh(arrow_from_force(point1, g_rot_1), color="green", **MESH_ARGS)
plotter.add_mesh(arrow_from_force(point1, g_climb_1), color="blue", **MESH_ARGS)


###### Frame of spin 2 ########
plotter.add_mesh(arrow_from_force(point2, force_1_rotated), color="black", **MESH_ARGS)
plotter.add_mesh(arrow_from_force(point2, force2), color="grey", **MESH_ARGS)

plotter.add_mesh(
    arrow_from_force(point2, g_orth_2, shaft_radius=SHAFT_RADIUS * 1.05),
    color="red",
    **MESH_ARGS,
)
plotter.add_mesh(arrow_from_force(point2, g_rot_2), color="green", **MESH_ARGS)
plotter.add_mesh(arrow_from_force(point2, g_climb_2), color="blue", **MESH_ARGS)

plotter.camera_position = "iso"
plotter.camera.view_angle /= 1.25

plotter.show(auto_close=False)
plotter.screenshot(os.path.join("spin_forces.png"))


logger.info(f"{point1 = }")
logger.info(f"{point2 = }")
logger.info(f"{point2 - point1 = }")
logger.info(f"{tangent_1 = }")
logger.info(f"{tangent_1.dot(point1) = }")
logger.info(f"{tangent_2 = }")
logger.info(f"{tangent_2.dot(point2) = }")
logger.info(f"{angle = }")
logger.info(f"{axis = }")
logger.info(f"{rotation_matrix.dot(point1) = }")
logger.info(f"{rotation_matrix.transpose().dot(point2) = }")
