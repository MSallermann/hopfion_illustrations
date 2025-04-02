import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Union
import pyvista as pv
import math
import enum
from pathlib import Path


def get_rgba_color(
    spin: npt.NDArray,
    opacity: float = 1.0,
    cardinal_a: npt.NDArray = np.array([1, 0, 0]),
    cardinal_b: npt.NDArray = np.array([0, 1, 0]),
    cardinal_c: npt.NDArray = np.array([0, 0, 1]),
) -> npt.NDArray:
    """
    Compute an RGBA color based on a 3D spin vector.

    The function projects the spin vector onto three cardinal axes and computes a hue from the
    arctangent of the projections in the plane defined by cardinal_a and cardinal_b. The saturation
    and value are derived from the projection on cardinal_c, and the resulting HSV color is converted
    to RGB and combined with the given opacity.

    Parameters:
        spin (npt.NDArray): A 3D vector representing the spin.
        opacity (float, optional): The opacity value for the color. Defaults to 1.0.
        cardinal_a (npt.NDArray, optional): First cardinal vector. Defaults to np.array([1, 0, 0]).
        cardinal_b (npt.NDArray, optional): Second cardinal vector. Defaults to np.array([0, 1, 0]).
        cardinal_c (npt.NDArray, optional): Third cardinal vector. Defaults to np.array([0, 0, 1]).

    Returns:
        npt.NDArray: A list of four values representing the RGBA color.
    """

    # Annoying OpenGl functions
    def atan2(y, x):
        if x == 0.0:
            return np.sign(y) * np.pi / 2.0
        else:
            return np.arctan2(y, x)

    def fract(x):
        return x - math.floor(x)

    def mix(x, y, a):
        return x * (1.0 - a) + y * a

    def clamp(x, minVal, maxVal):
        return min(max(x, minVal), maxVal)

    def hsv2rgb(c):
        K = [1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0]

        px = abs(fract(c[0] + K[0]) * 6.0 - K[3])
        py = abs(fract(c[0] + K[1]) * 6.0 - K[3])
        pz = abs(fract(c[0] + K[2]) * 6.0 - K[3])
        resx = c[2] * mix(K[0], clamp(px - K[0], 0.0, 1.0), c[1])
        resy = c[2] * mix(K[0], clamp(py - K[0], 0.0, 1.0), c[1])
        resz = c[2] * mix(K[0], clamp(pz - K[0], 0.0, 1.0), c[1])

        return [resx, resy, resz]

    projection_x = cardinal_a.dot(spin)
    projection_y = cardinal_b.dot(spin)
    projection_z = cardinal_c.dot(spin)
    hue = atan2(projection_x, projection_y) / (2 * np.pi)

    saturation = projection_z

    if saturation > 0.0:
        saturation = 1.0 - saturation
        value = 1.0
    else:
        value = 1.0 + saturation
        saturation = 1.0

    rgba = [*hsv2rgb([hue, saturation, value]), opacity]
    return rgba


def get_rgba_colors(
    spins: npt.NDArray,
    opacity: float = 1.0,
    cardinal_a: npt.NDArray = np.array([1, 0, 0]),
    cardinal_b: npt.NDArray = np.array([0, 1, 0]),
    cardinal_c: npt.NDArray = np.array([0, 0, 1]),
) -> npt.NDArray:
    """
    Compute RGBA colors for an array of spin vectors.

    This function applies the get_rgba_color function to each spin vector in the input array.

    Parameters:
        spins (npt.NDArray): An array of spin vectors.
        opacity (float, optional): The opacity value for all colors. Defaults to 1.0.
        cardinal_a (npt.NDArray, optional): First cardinal vector. Defaults to np.array([1, 0, 0]).
        cardinal_b (npt.NDArray, optional): Second cardinal vector. Defaults to np.array([0, 1, 0]).
        cardinal_c (npt.NDArray, optional): Third cardinal vector. Defaults to np.array([0, 0, 1]).

    Returns:
        npt.NDArray: An array of RGBA colors corresponding to the input spin vectors.
    """
    rgba = [
        get_rgba_color(s, opacity, cardinal_a, cardinal_b, cardinal_c) for s in spins
    ]
    return rgba


def line_from_3D_curve(curve_func: Callable, interval: npt.NDArray):
    """
    Generate a line from a 3D curve function.

    Samples the provided curve function at parameter values specified in 'interval'
    and constructs a PyVista line connecting these points.

    Parameters:
        curve_func (Callable): A function that takes a scalar t and returns a 3D point.
        interval (npt.NDArray): Array of parameter values to sample the curve.

    Returns:
        pv.PolyData: A PyVista PolyData object representing the line.
    """
    points = np.array([curve_func(t) for t in interval])
    return pv.lines_from_points(points)


def compute_frenet_frame_single(
    curve_func: Callable, t: float, epsilon: float = 1e-5
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compute the Frenet frame (T, N, B) at a specific parameter value on a curve.

    The function calculates the tangent (T), normal (N), and binormal (B) vectors
    using finite-difference approximations of the derivatives.

    Parameters:
        curve_func (Callable): A function that takes a scalar t and returns a 3D point.
        t (float): The parameter value at which to compute the Frenet frame.
        epsilon (float, optional): A small value used for finite-difference approximation. Defaults to 1e-5.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: A tuple containing the unit tangent (T),
        normal (N), and binormal (B) vectors.
    """
    # Approximate the derivative to get the tangent using central differences
    r_plus = np.array(curve_func(t + epsilon))
    r_minus = np.array(curve_func(t - epsilon))
    deriv = (r_plus - r_minus) / (2 * epsilon)
    T = deriv / np.linalg.norm(deriv)

    # Helper function to compute the tangent at an arbitrary parameter s
    def tangent_at(s):
        rp = np.array(curve_func(s + epsilon))
        rm = np.array(curve_func(s - epsilon))
        d = (rp - rm) / (2 * epsilon)
        return d / np.linalg.norm(d)

    # Approximate the derivative of T to get T'
    T_plus = tangent_at(t + epsilon)
    T_minus = tangent_at(t - epsilon)
    dT = (T_plus - T_minus) / (2 * epsilon)

    # Normalize dT to get the unit normal vector N
    norm_dT = np.linalg.norm(dT)
    if norm_dT < 1e-8:
        # If curvature is zero, the normal is undefined; return an arbitrary vector orthogonal to T.
        ez = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])
        N = np.cross(T, ez)
        N /= np.linalg.norm(N)
    else:
        N = dT / norm_dT

    # The binormal is the cross product of T and N
    B = np.cross(T, N)

    return T, N, B


def tube_from_3D_curve(
    curve_func: Callable,
    interval: npt.NDArray,
    radius: Union[Callable, float] = 1.0,
    resolution_phi: int = 32,
    epsilon: float = 1e-5,
    close_curve: bool = True,
    color_callback: Optional[Callable] = None,
) -> pv.PolyData:
    """
    Generate a tube mesh from a 3D curve.

    Constructs a tube by sweeping a circular cross-section along the curve defined by 'curve_func'.
    The orientation of the cross-section is determined by the Frenet frame computed at each sample point.

    Parameters:
        curve_func (Callable): A function that takes a scalar t and returns a 3D point.
        interval (npt.NDArray): Array of parameter values to sample the curve.
        radius (Union[Callable, float], optional): A constant radius or a function returning the radius at a given t. Defaults to 1.0.
        resolution_phi (int, optional): Number of divisions around the tube's circumference. Defaults to 32.
        epsilon (float, optional): A small value for finite-difference approximations. Defaults to 1e-5.
        close_curve (bool, optional): If True, the tube is closed along the curve. Defaults to True.
        color_callback (Optional[Callable], optional): Function to assign a color to each point. Defaults to None.

    Returns:
        pv.PolyData: A PyVista PolyData object representing the tube mesh.
    """
    points_curve = np.array([curve_func(t) for t in interval])

    points = []
    colors = []

    for t, p in zip(interval, points_curve):
        T, N, B = compute_frenet_frame_single(curve_func, t, epsilon)

        for phi in np.linspace(0, 2 * np.pi, resolution_phi, endpoint=False):
            if callable(radius):
                r = radius(t)
            else:
                r = radius
            point = p + r * (np.sin(phi) * N + np.cos(phi) * B)
            points.append(point)
            if color_callback is not None:
                colors.append(color_callback(point, t, phi))

    # Now we create rectangular faces by joining points along the circumference of the tube
    faces = []
    if close_curve:
        idx_curve_list = range(len(interval))
    else:
        idx_curve_list = range(len(interval) - 1)

    for idx_curve in idx_curve_list:
        for idx_phi in range(resolution_phi):
            idx_p1 = resolution_phi * idx_curve + idx_phi
            idx_p2 = resolution_phi * idx_curve + (idx_phi + 1) % resolution_phi
            idx_p3 = resolution_phi * ((idx_curve + 1) % len(interval)) + idx_phi
            idx_p4 = (
                resolution_phi * ((idx_curve + 1) % len(interval))
                + (idx_phi + 1) % resolution_phi
            )

            faces.append([4, idx_p1, idx_p3, idx_p4, idx_p2])

    res = pv.PolyData(points, faces=faces)
    if color_callback is not None:
        res.point_data["color"] = colors

    return res


def surface_from_equation(
    surface_func: Callable,
    interval_x: list[float],
    interval_y: list[float],
    color_callback: Optional[Callable] = None,
) -> pv.PolyData:
    points = []
    colors = []
    faces = []

    NX = len(interval_x)
    NY = len(interval_y)

    for ix, x in enumerate(interval_x):
        for iy, y in enumerate(interval_y):
            z = surface_func(x, y)
            points.append(np.array([x, y, z]))

            if color_callback is not None:
                colors.append(color_callback(x, y, z))

            if ix < NX - 1 and iy < NY - 1:
                idx_p1 = ix * NY + iy
                idx_p2 = (ix + 1) * NY + iy
                idx_p3 = (ix + 1) * NY + (iy + 1)
                idx_p4 = ix * NY + (iy + 1)
                faces.append([4, idx_p1, idx_p2, idx_p3, idx_p4])

    res = pv.PolyData(points, faces=faces)
    if color_callback is not None:
        res.point_data["color"] = colors

    return res


def preimage_from_3D_curve(
    curve_func: Callable,
    phi0: float,
    radius: Union[Callable, float],
    n_twists: int,
    epsilon: float = 1e-5,
) -> Callable:
    """
    Create a preimage function based on a 3D curve.

    This function generates a new curve by offsetting the original curve along its Frenet frame,
    applying an angular twist determined by 'n_twists' and an initial angle offset 'phi0'.

    Parameters:
        curve_func (Callable): A function that takes a scalar t and returns a 3D point.
        phi0 (float): Initial angular offset.
        radius (Union[Callable, float]): A constant radius or a function returning the radius at a given t.
        n_twists (int): Number of twists to apply along the curve.
        epsilon (float, optional): A small value for finite-difference approximations. Defaults to 1e-5.

    Returns:
        Callable: A function that maps a parameter t to a 3D point on the preimage curve.
    """

    def preimage(t):
        T, N, B = compute_frenet_frame_single(curve_func, t, epsilon)
        p = curve_func(t)
        phi = phi0 - n_twists * t
        if callable(radius):
            r = radius(t)
        else:
            r = radius
        point = p + r * (np.sin(phi) * N + np.cos(phi) * B)
        return point

    return preimage


def create_preimage_meshes(
    curve_func: Callable,
    phi_list: list[float],
    radius: Union[Callable, float],
    tube_radius: float,
    n_res_curve: int,
    n_twists: int,
    n_res_phi: int,
) -> list[pv.PolyData]:
    """
    Create tube meshes for preimage curves with different angular offsets.

    For each angle in 'phi_list', a preimage curve is generated and converted into a tube mesh.

    Parameters:
        curve_func (Callable): A function that takes a scalar t and returns a 3D point.
        phi_list (list[float]): List of initial angular offsets.
        radius (Union[Callable, float]): A constant radius or a function returning the radius at a given t.
        tube_radius (float): The radius of the tube around the preimage curve.
        n_res_curve (int): Number of sample points along the curve.
        n_twists (int): Number of twists to apply in generating the preimage curve.
        n_res_phi (int): Resolution (number of sides) of the tube's cross-section.

    Returns:
        list[pv.PolyData]: A list of PyVista PolyData objects representing the preimage tube meshes.
    """
    meshes = []

    interval = np.linspace(0, 2 * np.pi, n_res_curve, endpoint=True)

    for phi in phi_list:
        preimage_func = preimage_from_3D_curve(
            curve_func, phi0=phi, radius=radius, n_twists=n_twists
        )

        preimage_tube = line_from_3D_curve(
            preimage_func,
            interval=interval,
        ).tube(radius=tube_radius, n_sides=n_res_phi)

        meshes.append(preimage_tube)

    return meshes


def ring(t, radius=1):
    """
    Parametric equation for a ring (circle) in the XY-plane.

    Parameters:
        t (float): The parameter (angle in radians).
        radius (float, optional): The radius of the ring. Defaults to 1.

    Returns:
        list: A list containing the x, y, and z coordinates of the ring point.
    """
    return [np.cos(t) * radius, np.sin(t) * radius, 0]


def figure_eight(t):
    """
    Parametric equation for a figure-eight curve in the XY-plane.

    Parameters:
        t (float): The parameter (typically in radians).

    Returns:
        list: A list containing the x, y, and z coordinates of the figure-eight curve point.
    """
    return [np.sin(2.0 * t), np.cos(t), 0]


def trefoil(t):
    """
    Parametric equation for a trefoil knot.

    Parameters:
        t (float): The parameter (typically in radians).

    Returns:
        list: A list containing the x, y, and z coordinates of the trefoil knot point.
    """
    return [
        np.sin(t) + 2.0 * np.sin(2.0 * t),
        np.sin(3.0 * t),
        np.cos(t) - 2.0 * np.cos(2.0 * t),
    ]


class SkyBox(enum.Enum):
    black = "black"
    blue = "blue"
    dark_grey = "dark_grey"
    grey = "grey"
    light_grey = "light_grey"
    red = "red"
    very_light_grey = "very_light_grey"
    white = "white"
    yellow = "yellow"


def setup_plotter(
    offscreen: bool = False,
    resolution: tuple[float] = (2048, 720),
    skybox: SkyBox = SkyBox.very_light_grey,
):
    plotter = pv.Plotter(off_screen=offscreen, shape=(1, 1), window_size=resolution)

    # Create skybox
    image_paths = 6 * [f"{Path(__file__).parent}/cubemap/{skybox.value}.png"]
    cubemap = pv.cubemap_from_filenames(image_paths=image_paths)

    # Some plotter settings for "nice" looking renders
    plotter.enable_3_lights()
    plotter.set_environment_texture(cubemap)
    plotter.set_background("white")
    plotter.enable_anti_aliasing()
    plotter.enable_depth_peeling()
    plotter.enable_shadows()

    return plotter
