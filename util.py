import numpy as np
import numpy.typing as npt
from typing import Callable, Optional
import pyvista as pv
import math
from scipy.spatial.transform import Rotation as R


def get_rgba_color(
    spin: npt.NDArray,
    opacity: float = 1.0,
    cardinal_a: npt.NDArray = np.array([1, 0, 0]),
    cardinal_b: npt.NDArray = np.array([0, 1, 0]),
    cardinal_c: npt.NDArray = np.array([0, 0, 1]),
) -> npt.NDArray:

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

    rgba = [
        get_rgba_color(s, opacity, cardinal_a, cardinal_b, cardinal_c) for s in spins
    ]
    return rgba


def line_from_3D_curve(curve_func: Callable, interval: npt.NDArray):
    points = np.array([curve_func(t) for t in interval])
    return pv.lines_from_points(points)


def compute_frenet_frame_single(curve_func, t, epsilon=1e-5):
    """
    Compute the Frenet frame (point, T, N, B) at a single parameter value t.

    Parameters:
      curve_func: callable
          A function that takes a scalar t and returns a 3D point (as a list or numpy array).
      t: float
          The parameter value at which to compute the Frenet frame.
      epsilon: float
          A small value used for finite-difference derivative approximations.

    Returns:
      point: np.ndarray
          The point on the curve at parameter t.
      T: np.ndarray
          The unit tangent vector at t.
      N: np.ndarray
          The unit normal vector at t.
      B: np.ndarray
          The binormal vector at t.
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
        # If curvature is zero, the normal is undefined; here we return an arbitrary vector orthogonal to the tangent
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
    radius: float = 1.0,
    resolution_phi: int = 32,
    epsilon: float = 1e-7,
    close_curve: bool = True,
    color_callback: Optional[Callable] = None,
):

    points_curve = np.array([curve_func(t) for t in interval])

    # this is an arbitrary helper vector

    points = []
    colors = []

    for t, p in zip(interval, points_curve):
        T, N, B = compute_frenet_frame_single(curve_func, t, epsilon)

        for phi in np.linspace(0, 2 * np.pi, resolution_phi, endpoint=False):
            point = p + radius * (np.sin(phi) * N + np.cos(phi) * B)
            points.append(point)
            if not color_callback is None:
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
    if not color_callback is None:
        res.point_data["color"] = colors

    return res


def ring(t, radius=1):
    return [np.cos(t) * radius, np.sin(t) * radius, 0]


def figure_eight(t):
    return [np.sin(2.0 * t), np.cos(t), 0]


def trefoil(t):
    return [
        np.sin(t) + 2.0 * np.sin(2.0 * t),
        np.sin(3.0 * t),
        np.cos(t) - 2.0 * np.cos(2.0 * t),
    ]
