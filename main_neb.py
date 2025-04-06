import numpy as np
import numpy.typing as npt
import util
from energy_surfaces.energy_surface import EnergySurface
from energy_surfaces.surfaces import cosine_ssbench
import pyvista as pv
from ridgefollowing.algorithms import minimizer, neb
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
log_file = Path(__file__).with_suffix(".log")
logging.basicConfig(filename=log_file, level=logging.INFO)

# ########################################
#           SETUP
# ########################################

OFFSCREEN = True
plotter = util.setup_plotter(
    offscreen=OFFSCREEN, skybox=util.SkyBox.light_grey, resolution=[1524, 720]
)

N_RES_X = 200
N_RES_Y = 200
N_RES_PATH = 100


MESH_ARGS = {
    "rgb": False,
    "smooth_shading": True,
    "pbr": True,
    "metallic": 0.55,
    "roughness": 0.45,
    "cmap": "viridis",
    "scalars": "energy",
    "line_width": 1.0,
    "show_scalar_bar": False,
}

# ########################################
#          Define the surface
# ########################################

# surf = peaks.PeaksSurface()

# energy_surface = lepshogauss.LepsHOGaussSurface()
# INTERVAL_X = np.linspace(0.55, 3.5, dtype=float)
# INTERVAL_Y = np.linspace(-2.0, 5.0, dtype=float)

# X_START_INIT = np.array([0.6759588922580516, 4.271771599520583])
# # X_END_INIT = np.array([0.7834647677581984, -0.6289325321256285])
# X_END_INIT = np.array([3.0012763798644935, -1.304342056715238 ])

energy_surface = cosine_ssbench.CosineSSBENCH()
INTERVAL_X = np.linspace(-6.0, 9.0, dtype=float, num=N_RES_X)
INTERVAL_Y = np.linspace(-1.0, 12.0, dtype=float, num=N_RES_Y)
X_START_INIT = np.array([3.32099292, 4.24204202])
X_END_INIT = np.array([5.46691415e-03, 6.28604213e00])


def surface_func(x, y):
    return energy_surface.energy(np.array([x, y]))


# ########################################
#        Find start and end point
# ########################################

min = minimizer.Minimizer(energy_surface=energy_surface, tolerance=1e-4)

x_start = min.minimize_energy(X_START_INIT)
logger.info(f"{x_start = }")

x_end = min.minimize_energy(X_END_INIT)
logger.info(f"{x_end = }")


# ########################################
#        Find the MEP
# ########################################

path_images = Path("./images.npy")
path_rx = Path("./rx.npy")

if path_images.exists() and path_rx.exists():
    images, rx = np.load(path_images), np.load(path_rx)
else:
    neb_run = neb.GNEB(
        energy_surface=energy_surface,
        x_start=x_start,
        x_end=x_end,
        num_images=24,
        step_size=1e-2,
        convergence_tol=1e-4,
        max_iter=10000,
    )
    neb_run.run()
    images, rx = neb_run.get_path()
    np.save(path_images, images)
    np.save(path_rx, rx)


# ########################################
#          Plot the surface
# ########################################

surf = util.surface_from_equation(
    surface_func=surface_func, interval_x=INTERVAL_X, interval_y=INTERVAL_Y
)
surf["energy"] = surf.points[:, 2]

plotter.add_mesh(surf, **MESH_ARGS)

# ########################################
#          Plot the path
# ########################################


def mark_point(
    plotter: pv.Plotter,
    p: npt.ArrayLike,
    color: str,
    radius: float = 0.05,
    mesh_args=MESH_ARGS,
):
    x, y = p
    sph = pv.Sphere(radius=radius, center=[x, y, surface_func(x, y)])
    mesh_args = mesh_args.copy()
    mesh_args["rgb"] = False
    mesh_args["color"] = color
    mesh_args["scalars"] = None
    plotter.add_mesh(sph, **mesh_args)


def mark_path(
    plotter: pv.Plotter,
    rx: list[float],
    images: list[npt.ArrayLike],
    energy_surface: EnergySurface,
    color_endpoints: str = "black",
    color_tube: str = "black",
    color_images: str = "grey",
    color_sp: str = "red",
    radius_point_spheres: float = 0.05,
    tube_radius_path: float = 0.025,
    n_res_path: int = N_RES_PATH,
    mesh_args: dict = MESH_ARGS,
    mark_intermediate_points: bool = True,
    mark_first_point: bool = True,
    mark_last_point: bool = True,
    mark_saddle_point: bool = True,
    mark_interpolated_path: bool = True,
):
    energy_images = [energy_surface.energy(i) for i in images]
    idx_sp = np.argmax(energy_images)
    sp = images[idx_sp]

    if mark_intermediate_points:
        [
            mark_point(
                plotter,
                img,
                color=color_images,
                radius=radius_point_spheres,
                mesh_args=mesh_args,
            )
            for img in images[1:-1]
        ]

    if mark_saddle_point:
        mark_point(
            plotter,
            sp,
            color=color_sp,
            radius=radius_point_spheres,
            mesh_args=mesh_args,
        )

    if mark_first_point:
        mark_point(
            plotter,
            images[0],
            color_endpoints,
            radius=radius_point_spheres,
            mesh_args=mesh_args,
        )

    if mark_last_point:
        mark_point(
            plotter,
            images[-1],
            color_endpoints,
            radius=radius_point_spheres,
            mesh_args=mesh_args,
        )

    grad_sp = energy_surface.gradient(sp)
    logger.info(f"{grad_sp=}")

    hess_sp = energy_surface.hessian(sp)
    logger.info(f"{hess_sp = }")
    logger.info(f"{np.linalg.eigh(hess_sp) = }")

    parametrized_path = util.parametrized_path(images, rx)

    def curve_func(t):
        point = parametrized_path(t)
        energy = surface_func(point[0], point[1])
        return [point[0], point[1], energy]

    logger.info(images)
    logger.info(rx)

    if mark_interpolated_path:
        path_as_tube = util.line_from_3D_curve(
            curve_func, interval=np.linspace(rx[0], rx[-1], n_res_path)
        )
        path_as_tube["energy"] = path_as_tube.points[:, 2]
        path_as_tube.tube(radius=tube_radius_path, inplace=True)

        mesh_args = mesh_args.copy()
        mesh_args["rgb"] = False
        mesh_args["color"] = color_tube
        mesh_args["scalars"] = None
        mesh_args["pbr"] = False

        plotter.add_mesh(path_as_tube, **mesh_args)

    return curve_func, idx_sp


path_curve_func, idx_sp = mark_path(
    plotter, rx, images, energy_surface=energy_surface
)


# ########################################
#          Finalize
# ########################################

cpos = [
    (1.461096353582104, 9.954686329132304, 5.461085853395577),
    (1.6715340367181246, 5.589868448110303, -0.2516462319861352),
    (-0.008120532699394296, -0.7947270023077835, 0.6069127192204515),
]

cpos = [
    (-5.01971218986893, 1.9731464297155148, 5.273052913308826),
    (2.0762806876502142, 5.496551693411181, -1.6640199437409793),
    (0.5790219711014105, 0.31470305680146626, 0.7521273449500758),
]

plotter.camera.position = cpos[0]
plotter.camera.focal_point = cpos[1]
plotter.camera.view_up = cpos[2]
plotter.camera.view_angle /= 1.5

cpos = plotter.show(return_cpos=True, screenshot="gneb.png")
logger.info(f"{cpos = }")
