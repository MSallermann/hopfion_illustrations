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
plotter.disable_shadows()

N_RES_X = 100
N_RES_Y = 100
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

from main_neb import mark_path

path_curve_func, idx_sp = mark_path(
    plotter, rx, images, energy_surface=energy_surface, mark_intermediate_points=False, mark_interpolated_path=False
)

# ########################################
#           Mark transition state
# ########################################

sp_xy = images[idx_sp]
energy_sp = energy_surface.energy(sp_xy)
tangent_sp = images[idx_sp + 1] - images[idx_sp - 1]
hessian_sp = energy_surface.hessian(sp_xy)

transition_plane = pv.Plane(
    center=[*images[idx_sp], energy_sp],
    direction=[*tangent_sp, 0],
    i_size=10,
    j_size=10,
    i_resolution=40,
    j_resolution=40
)
plotter.add_mesh(transition_plane, opacity=0.5, color="white", show_edges=True)


def parabola_around_point(
    point_xy,
    energy_surface,
    interval_x=np.linspace(-1, 1, 300),
    interval_y=np.linspace(-1, 1, 300),
):
    energy = energy_surface.energy(point_xy)
    grad = energy_surface.gradient(point_xy)
    hessian = energy_surface.hessian(point_xy)

    def surface_func_parabola(x, y):
        v = np.array([x, y]) - point_xy
        return 0.5 * v.T @ hessian @ v + grad @ v + energy

    mesh_parabola = util.surface_from_equation(
        surface_func=surface_func_parabola,
        interval_x=point_xy[0] + interval_x,
        interval_y=point_xy[1] + interval_y,
    )

    radius = 0.65
    mask = np.linalg.norm(mesh_parabola.points[:, :2] - point_xy, axis=1) < radius
    mesh = mesh_parabola.extract_points(mask)

    return mesh


def eigenvector_parabolas_around_point(
    point_xy,
    interval: float,
    tube_radius: float,
    energy_surface,
):
    energy = energy_surface.energy(point_xy)
    grad = energy_surface.gradient(point_xy)
    hessian = energy_surface.hessian(point_xy)

    evals, evecs = np.linalg.eigh(hessian)
    logger.info(evals)
    logger.info(evecs)

    def par(t, dir):
        p = point_xy + t * dir
        return [*p, energy + grad.T @ dir + 0.5 * t**2 * dir.T @ hessian @ dir]

    par_mesh_1 = util.tube_from_3D_curve(
        lambda t: par(t, evecs[:, 0]),
        interval=interval,
        radius=tube_radius,
        close_curve=False,
    )
    par_mesh_2 = util.tube_from_3D_curve(
        lambda t: par(t, evecs[:, 1]),
        interval=interval,
        radius=tube_radius,
        close_curve=False,
    )

    return par_mesh_1, par_mesh_2


## parabola around SP
mesh_parabola_sp = parabola_around_point(
    point_xy=images[idx_sp], energy_surface=energy_surface
).translate([0, 0, 3e-2], inplace=True)
plotter.add_mesh(mesh_parabola_sp, smooth_shading=True, color="red", opacity=0.25)

## parabola around Min
mesh_parabola_min = parabola_around_point(
    point_xy=images[-1], energy_surface=energy_surface
).translate([0, 0, 3e-2], inplace=True)
plotter.add_mesh(mesh_parabola_min, smooth_shading=True, color="green", opacity=0.25)

par_min1, par_min2 = eigenvector_parabolas_around_point(
    point_xy=images[-1],
    energy_surface=energy_surface,
    interval=np.linspace(-0.65, 0.65, 100),
    tube_radius=0.020,
)
plotter.add_mesh(par_min1, smooth_shading=True, color="darkgreen")
plotter.add_mesh(par_min2, smooth_shading=True, color="darkgreen")

par_sp1, par_sp2 = eigenvector_parabolas_around_point(
    point_xy=images[idx_sp],
    energy_surface=energy_surface,
    interval=np.linspace(-1, 1, 100),
    tube_radius=0.020,
)
plotter.add_mesh(par_sp1, smooth_shading=True, color="darkred")
plotter.add_mesh(par_sp2, smooth_shading=True, color="darkgreen")


def mark_perpendicular_velocities(plotter, point_xy, arrow_scale=-0.5, num_arrows=10):
    energy = energy_surface.energy(point_xy)
    grad = energy_surface.gradient(point_xy)
    hessian = energy_surface.hessian(point_xy)

    evals, evecs = np.linalg.eigh(hessian)

    def par(t, dir):
        p = point_xy + t * dir
        return [*p, energy + grad.T @ dir + 0.5 * t**2 * dir.T @ hessian @ dir]

    for t in np.linspace(-1, 0, endpoint=True, num=num_arrows):
        start = par(t, dir=evecs[:, 1])
        direction = -np.array([*evecs[:, 0], 0])
        arr = pv.Arrow(
            start=start, direction=-direction, scale=arrow_scale * np.linalg.norm(t)
        )
        plotter.add_mesh(arr, color="black")


mark_perpendicular_velocities(plotter, images[idx_sp])

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

cpos = plotter.show(return_cpos=True, screenshot="htst.png")
logger.info(f"{cpos = }")
