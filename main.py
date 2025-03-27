import pyvista as pv
import numpy as np
import util

plotter = pv.Plotter(off_screen=False, shape=(1, 1))
plotter.window_size = (1000, 1000)

# Create white skybox
image_paths = 6 * ["./cubemap/very_light_grey.png"]
cubemap = pv.cubemap_from_filenames(image_paths=image_paths)

# Some plotter settings
plotter.enable_3_lights()
plotter.set_environment_texture(cubemap)
plotter.set_background("white")
plotter.enable_anti_aliasing()
plotter.enable_depth_peeling()
plotter.enable_shadows()


N_RES_CURVE = 500
N_RES_PHI = 64

## For quicker testing
# N_RES_CURVE = 50
# N_RES_PHI = 16

MESH_ARGS = {
    "rgb": True,
    "smooth_shading": True,
    "pbr": True,
    "metallic": 0.75,
    "roughness": 0.15,
}


def color_callback(point, t, phi, n_twists):
    color = util.get_rgba_color(
        np.array([np.cos(n_twists * t + phi), np.sin(n_twists * t + phi), 0])
    )
    return color


# ########################################
#           Trefoil Hopfion
# ########################################
trefoil_hopfion = util.tube_from_3D_curve(
    util.trefoil,
    interval=np.linspace(0, 2 * np.pi, N_RES_CURVE, endpoint=False),
    radius=0.5,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi: color_callback(point, t, phi, n_twists=1),
)
plotter.add_mesh(trefoil_hopfion, **MESH_ARGS)

# ########################################
#         Toroidal Hopfion
# ########################################
toroidal_hopfion = util.tube_from_3D_curve(
    lambda t: util.ring(t, radius=1.5),
    interval=np.linspace(0, 2 * np.pi, N_RES_CURVE),
    radius=0.75,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi: color_callback(point, t, phi, n_twists=6),
)
toroidal_hopfion.translate([7, 0, 0], inplace=True)
plotter.add_mesh(toroidal_hopfion, **MESH_ARGS)


########################################
#        Skyrmion tube and Hopfion
########################################
toroidal_hopfion_tube = util.tube_from_3D_curve(
    lambda t: util.ring(t, radius=3),
    interval=np.linspace(0, 2 * np.pi, N_RES_CURVE, endpoint=False),
    radius=0.75,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi: color_callback(point, t, phi, n_twists=1),
)
toroidal_hopfion_tube.translate([-7, 0, 0], inplace=True)
plotter.add_mesh(toroidal_hopfion_tube, **MESH_ARGS)


sk_tube = util.tube_from_3D_curve(
    lambda t: [0, 0, t],
    interval=np.linspace(-3, 3, N_RES_CURVE, endpoint=True),
    radius=1.5,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi: util.get_rgba_color(
        np.array([np.cos(phi), np.sin(phi), 0])
    ),
    close_curve=False,
)
sk_tube.translate([-7, 0, 0], inplace=True)
plotter.add_mesh(sk_tube, **MESH_ARGS)

plotter.show(screenshot="screen.png")
