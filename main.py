import pyvista as pv
import numpy as np
import util

# ########################################
#           SETUP
# ########################################

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

# For quicker testing
N_RES_CURVE = 50
N_RES_PHI = 16

INTERVAL = np.linspace(0, 2 * np.pi, N_RES_CURVE, endpoint=False)

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


PREIMAGE_PHI_LIST = np.linspace(0, 2 * np.pi, 16, endpoint=False)


def render_hopfion_pre_images(
    plotter,
    phi_list,
    curve_func,
    radius,
    n_twists,
    translate_vector=np.zeros(3),
    radius_factor=1.125,
    tube_radius=0.045,
):
    pre_images = util.create_preimage_meshes(
        curve_func,
        phi_list=phi_list,
        radius=radius * radius_factor,
        tube_radius=tube_radius,
        n_res_curve=N_RES_CURVE,
        n_twists=n_twists,
        n_res_phi=N_RES_PHI,
    )

    [p.translate(translate_vector, inplace=True) for p in pre_images]

    for phi, m in zip(phi_list, pre_images):
        args = MESH_ARGS.copy()
        args["rgb"] = False
        args["color"] = util.get_rgba_color(np.array([np.cos(phi), np.sin(phi), 0]))
        plotter.add_mesh(m, **args)


# ########################################
#           Trefoil Hopfion
# ########################################


def plot_trefoil_hopfion(plotter, tube_radius, n_twists, translate_vector, pre_images):

    curve_func = util.trefoil

    trefoil_hopfion = util.tube_from_3D_curve(
        curve_func,
        interval=INTERVAL,
        radius=tube_radius,
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: color_callback(
            point, t, phi, n_twists=n_twists
        ),
    ).translate(translate_vector, inplace=True)
    plotter.add_mesh(trefoil_hopfion, **MESH_ARGS)

    if pre_images:
        render_hopfion_pre_images(
            plotter,
            PREIMAGE_PHI_LIST,
            curve_func=curve_func,
            radius=tube_radius,
            n_twists=n_twists,
            translate_vector=translate_vector,
        )


# ########################################
#         Toroidal Hopfion
# ########################################


def plot_toroidal_hopfion(
    plotter, n_twists, ring_radius, tube_radius, translate_vector, pre_images
):
    curve_func = lambda t: util.ring(t, radius=ring_radius)

    toroidal_hopfion = util.tube_from_3D_curve(
        curve_func,
        interval=INTERVAL,
        radius=tube_radius,
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: color_callback(
            point, t, phi, n_twists=n_twists
        ),
    ).translate(translate_vector, inplace=True)
    plotter.add_mesh(toroidal_hopfion, **MESH_ARGS)

    if pre_images:
        render_hopfion_pre_images(
            plotter,
            PREIMAGE_PHI_LIST,
            curve_func=curve_func,
            radius=tube_radius,
            n_twists=n_twists,
            translate_vector=translate_vector,
        )


########################################
#        Skyrmion tube
########################################


def plot_skyrmion_tube(plotter, radius, height_start, height_end, translate_vector):
    sk_tube = util.tube_from_3D_curve(
        lambda t: [0, 0, t],
        interval=np.linspace(height_start, height_end, N_RES_CURVE, endpoint=True),
        radius=radius,
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: util.get_rgba_color(
            np.array([np.cos(phi), np.sin(phi), 0])
        ),
        close_curve=False,
    )
    sk_tube.translate(translate_vector, inplace=True)
    plotter.add_mesh(sk_tube, **MESH_ARGS)


########################################
#        Bobber
########################################
def plot_bobber(plotter, height_start, height_end, radius_end, translate_vector):

    alpha = radius_end / np.sqrt(height_end - height_start)

    bobber = util.tube_from_3D_curve(
        lambda t: [0, 0, t],
        interval=np.linspace(height_start, height_end, N_RES_CURVE, endpoint=True),
        radius=lambda t, alpha=alpha: alpha * np.sqrt(t - height_start + 1e-8),
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: util.get_rgba_color(
            np.array([np.cos(phi), np.sin(phi), 0])
        ),
        close_curve=False,
    )
    bobber.translate(translate_vector, inplace=True)
    plotter.add_mesh(bobber, **MESH_ARGS)


# ########################################
#                 PLOT
# ########################################

# Single skyrmion tube
plot_skyrmion_tube(
    plotter, radius=1.0, height_start=-3, height_end=3, translate_vector=[-14, 0, 0]
)

# Single bobber
plot_bobber(
    plotter, height_start=0, height_end=3, radius_end=1.5, translate_vector=[-7, 0, 0]
)

# Trefoil Hopfion
plot_trefoil_hopfion(
    plotter, tube_radius=0.5, n_twists=2, translate_vector=[0, 0, 0], pre_images=True
)

# Toroidal Hopfion
plot_toroidal_hopfion(
    plotter,
    ring_radius=2.0,
    tube_radius=0.5,
    n_twists=2,
    translate_vector=[7, 0, 0],
    pre_images=True,
)

# Toroidal Hopfion on a skyrmion tube
plot_toroidal_hopfion(
    plotter,
    ring_radius=2.0,
    tube_radius=0.5,
    n_twists=1,
    translate_vector=[14, 0, 0],
    pre_images=True,
)
plot_skyrmion_tube(
    plotter, radius=1.0, height_start=-3, height_end=3, translate_vector=[14, 0, 0]
)

plotter.show(screenshot="screen.png")
