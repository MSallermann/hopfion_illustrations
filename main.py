import pyvista as pv
import numpy as np
import util
import link

# ########################################
#           SETUP
# ########################################

OFFSCREEN = True
plotter = pv.Plotter(off_screen=OFFSCREEN, shape=(1, 1))
plotter.window_size = (2048, 720)

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


N_RES_CURVE = 2000
N_RES_PHI = 128

# For quicker testing
# N_RES_CURVE = 50
# N_RES_PHI = 16

INTERVAL = np.linspace(0, 2 * np.pi, N_RES_CURVE, endpoint=False)

N_PREIMAGES = 24

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

    pre_image_1 = util.preimage_from_3D_curve(
        curve_func, phi0=0, radius=radius, n_twists=n_twists
    )
    pre_image_2 = util.preimage_from_3D_curve(
        curve_func, phi0=np.pi, radius=radius, n_twists=n_twists
    )

    linking_number = link.compute_linking_number(
        pre_image_1,
        pre_image_2,
        num_points=N_RES_CURVE,
        t_range=(INTERVAL[0], INTERVAL[-1]),
    )

    print(f"{n_twists = }")
    print(f"{linking_number = }")

    [p.translate(translate_vector, inplace=True) for p in pre_images]

    for phi, m in zip(phi_list, pre_images):
        args = MESH_ARGS.copy()
        args["rgb"] = False
        args["color"] = util.get_rgba_color(np.array([np.cos(phi), np.sin(phi), 0]))
        plotter.add_mesh(m, **args)


# ########################################
#           Trefoil Hopfion
# ########################################


def plot_trefoil_hopfion(
    plotter,
    tube_radius,
    n_twists,
    translate_vector,
    pre_images,
    n_preimages=N_PREIMAGES,
    mesh_args=MESH_ARGS,
):
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
    plotter.add_mesh(trefoil_hopfion, **mesh_args)

    PREIMAGE_PHI_LIST = np.linspace(0, 2 * np.pi, n_preimages, endpoint=False)

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
    plotter,
    n_twists,
    ring_radius,
    tube_radius,
    translate_vector,
    pre_images,
    n_preimages=N_PREIMAGES,
    mesh_args=MESH_ARGS,
):
    def curve_func(t):
        return util.ring(t, radius=ring_radius)

    toroidal_hopfion = util.tube_from_3D_curve(
        curve_func,
        interval=INTERVAL,
        radius=tube_radius,
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: color_callback(
            point, t, phi, n_twists=n_twists
        ),
    ).translate(translate_vector, inplace=True)
    plotter.add_mesh(toroidal_hopfion, **mesh_args)

    PREIMAGE_PHI_LIST = np.linspace(0, 2 * np.pi, n_preimages, endpoint=False)

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


def plot_skyrmion_tube(
    plotter, radius, height_start, height_end, translate_vector, mesh_args=MESH_ARGS
):
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
    plotter.add_mesh(sk_tube, **mesh_args)


########################################
#        Bobber
########################################
def plot_bobber(
    plotter, height_start, height_end, radius_end, translate_vector, mesh_args=MESH_ARGS
):
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
    plotter.add_mesh(bobber, **mesh_args)


########################################
#        Globule
########################################
def plot_globule(
    plotter,
    height_start,
    height_end,
    translate_vector,
    radius_center=None,
    mesh_args=MESH_ARGS,
):
    R = (height_end - height_start) / 2.0
    t_mid = (height_end + height_start) / 2.0

    if radius_center is not None:
        alpha = radius_center / R
    else:
        alpha = 1.0

    globule = util.tube_from_3D_curve(
        lambda t: [0, 0, t],
        interval=np.linspace(height_start, height_end, N_RES_CURVE, endpoint=True),
        radius=lambda t: alpha * np.sqrt(R**2 - (t - t_mid) ** 2 + 1e-8),
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: util.get_rgba_color(
            np.array([np.cos(phi), np.sin(phi), 0])
        ),
        close_curve=False,
    )
    globule.translate(translate_vector, inplace=True)
    plotter.add_mesh(globule, **mesh_args)


# ########################################
#                 PLOT
# ########################################

# Single skyrmion tube
plot_skyrmion_tube(
    plotter, radius=1.0, height_start=-3, height_end=3, translate_vector=[-5, 0, 0]
)

# Single bobber
plot_bobber(
    plotter, height_start=0, height_end=3, radius_end=1.5, translate_vector=[0, 0, 0]
)

plot_globule(
    plotter,
    height_start=-1,
    height_end=1,
    radius_center=1.2,
    translate_vector=[5, 0, 0],
)

# # Trefoil Hopfion
# plot_trefoil_hopfion(
#     plotter, tube_radius=0.5, n_twists=-3, translate_vector=[0, 0, 0], pre_images=True
# )

# # Toroidal Hopfion
# plot_toroidal_hopfion(
#     plotter,
#     ring_radius=2.0,
#     tube_radius=0.5,
#     n_twists=6,
#     translate_vector=[7, 0, 0],
#     pre_images=True,
# )

# Toroidal Hopfion on a skyrmion tube
mesh_args = MESH_ARGS.copy()
mesh_args.update({"rgb": False, "color": "white"})
plot_toroidal_hopfion(
    plotter,
    ring_radius=2.0,
    tube_radius=0.5,
    n_twists=1,
    translate_vector=[10, 0, 0],
    pre_images=True,
    mesh_args=mesh_args,
)
plot_skyrmion_tube(
    plotter, radius=1.0, height_start=-3, height_end=3, translate_vector=[10, 0, 0]
)

plotter.camera_position = "xz"
plotter.camera.elevation = 20
plotter.camera.distance = 0.5
plotter.camera.view_angle /= 2.2

cpos = plotter.show(screenshot="figure.png", return_cpos=True)
