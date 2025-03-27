import pyvista as pv
import numpy as np
import util

plotter = pv.Plotter(off_screen=False, shape=(1, 1))
plotter.window_size = (2048, 2048)
# Create white skybox
image_paths = 6 * ["./cubemap/white.png"]
cubemap = pv.cubemap_from_filenames(image_paths=image_paths)
plotter.enable_3_lights()
plotter.set_environment_texture(cubemap)
plotter.set_background("white")

N_RES_CURVE = 500
N_RES_PHI = 64

# N_RES_CURVE = 50
# N_RES_PHI = 16

def color_callback(point, t, phi, n_twists):
    color = util.get_rgba_color(np.array([np.cos(n_twists*t + phi), np.sin(n_twists*t + phi), 0]))
    return color

# ########################################
# #           Trefoil Hopfion
# ########################################
trefoil_hopfion = util.tube_from_3D_curve(
    util.trefoil,
    interval=np.linspace(0, 2 * np.pi, N_RES_CURVE),
    radius=0.5,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi : color_callback(point, t, phi, n_twists=1),
)
plotter.add_mesh(trefoil_hopfion, rgb=True, smooth_shading=True, pbr=True, metallic=0.65, roughness=0.45, opacity=None)

# ########################################
# #           Toroidal Hopfion
# ########################################
toroidal_hopfion = util.tube_from_3D_curve(
    lambda t : util.ring(t, radius=1.5),
    interval=np.linspace(0, 2 * np.pi, N_RES_CURVE),
    radius=0.75,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi : color_callback(point, t, phi, n_twists=6),
)
toroidal_hopfion.translate([7,0,0], inplace=True)
plotter.add_mesh(toroidal_hopfion, rgb=True, smooth_shading=True, pbr=True, metallic=0.65, roughness=0.45, opacity=None)


########################################
#           Skyrmion tube and Hopfion
########################################
toroidal_hopfion_tube = util.tube_from_3D_curve(
    lambda t : util.ring(t, radius=3),
    interval=np.linspace(0, 2 * np.pi, N_RES_CURVE),
    radius=0.75,
    resolution_phi=N_RES_PHI,
    color_callback=lambda point, t, phi : color_callback(point, t, phi, n_twists=1)
)
toroidal_hopfion_tube.translate([-7,0,0], inplace=True)
plotter.add_mesh(toroidal_hopfion_tube, rgb=True, smooth_shading=True, pbr=True, metallic=0.65, roughness=0.45, opacity=None)


sk_tube = util.tube_from_3D_curve(
    lambda t: [0,0,t],
    interval=np.linspace(-3,3, N_RES_CURVE),
    radius = 1.5,
    resolution_phi=N_RES_PHI,
    color_callback= lambda point, t, phi : util.get_rgba_color(np.array([np.cos(phi), np.sin( phi), 0])),
    close_curve=False
)
sk_tube.translate([-7,0,0], inplace=True)
plotter.add_mesh(sk_tube, rgb=True, smooth_shading=True, pbr=True, metallic=0.65, roughness=0.45, opacity=None)


plotter.show(screenshot="screen.png")
