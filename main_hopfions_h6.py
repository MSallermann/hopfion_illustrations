"""Render the two charge-six Hopfions shown in ``res/hopfions.png``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pyvista as pv

import util

N_RES_CURVE = 2000
N_RES_PHI = 128
N_PREIMAGES = 24

INTERVAL = np.linspace(0, 2 * np.pi, N_RES_CURVE, endpoint=False)

MESH_ARGS = {
    "smooth_shading": True,
    "pbr": True,
    "metallic": 0.75,
    "roughness": 0.15,
}

HOPFION_MESH_ARGS = {
    **MESH_ARGS,
    "rgb": False,
    "color": "grey",
}


def setup_plotter() -> pv.Plotter:
    """Recreate the lighting and 2048 x 720 canvas used by the original."""
    plotter = pv.Plotter(off_screen=True, shape=(1, 1), window_size=(2048 * 2, 720 * 2))

    cubemap_path = Path(__file__).parent / "cubemap" / "very_light_grey.png"
    cubemap = pv.cubemap_from_filenames([str(cubemap_path)] * 6)

    plotter.enable_3_lights()
    plotter.set_environment_texture(cubemap)
    plotter.set_background("white")
    plotter.enable_anti_aliasing()
    plotter.enable_depth_peeling()
    plotter.enable_shadows()
    return plotter


def color_callback(
    _point: np.ndarray, t: float, phi: float, n_twists: int
) -> np.ndarray:
    return util.get_rgba_color(
        np.array([np.cos(n_twists * t + phi), np.sin(n_twists * t + phi), 0])
    )


def add_preimages(
    plotter: pv.Plotter,
    curve_func: Callable[[float], list[float] | np.ndarray],
    radius: float,
    n_twists: int,
    translate_vector: list[float],
    radius_factor: float = 1.125,
    tube_radius: float = 0.045,
) -> None:
    phi_list = np.linspace(0, 2 * np.pi, N_PREIMAGES, endpoint=False)
    preimages = util.create_preimage_meshes(
        curve_func,
        phi_list=phi_list,
        radius=radius * radius_factor,
        tube_radius=tube_radius,
        n_res_curve=N_RES_CURVE,
        n_twists=n_twists,
        n_res_phi=N_RES_PHI,
    )

    for phi, mesh in zip(phi_list, preimages):
        mesh.translate(translate_vector, inplace=True)
        mesh_args = MESH_ARGS.copy()
        mesh_args["rgb"] = False
        mesh_args["color"] = util.get_rgba_color(
            np.array([np.cos(phi), np.sin(phi), 0])
        )
        plotter.add_mesh(mesh, **mesh_args)


def add_hopfion(
    plotter: pv.Plotter,
    curve_func: Callable[[float], list[float] | np.ndarray],
    tube_radius: float,
    n_twists: int,
    translate_vector: list[float],
) -> None:
    hopfion = util.tube_from_3D_curve(
        curve_func,
        interval=INTERVAL,
        radius=tube_radius,
        resolution_phi=N_RES_PHI,
        color_callback=lambda point, t, phi: color_callback(point, t, phi, n_twists),
    )
    hopfion.translate(translate_vector, inplace=True)
    plotter.add_mesh(hopfion, **HOPFION_MESH_ARGS)

    add_preimages(
        plotter,
        curve_func=curve_func,
        radius=tube_radius,
        n_twists=n_twists,
        translate_vector=translate_vector,
    )


def render(output: Path) -> None:
    plotter = setup_plotter()

    # Trefoil Hopfion with linking number H = 6.
    add_hopfion(
        plotter,
        curve_func=util.trefoil,
        tube_radius=0.5,
        n_twists=-3,
        translate_vector=[0, 0, 0],
    )

    # Toroidal Hopfion with linking number H = 6.
    def toroidal_curve(t: float) -> list[float]:
        return util.ring(t, radius=2.0)

    add_hopfion(
        plotter,
        curve_func=toroidal_curve,
        tube_radius=0.5,
        n_twists=6,
        translate_vector=[7, 0, 0],
    )

    plotter.camera_position = "xz"
    plotter.camera.elevation = 35
    plotter.camera.distance = 0.5
    plotter.camera.view_angle /= 2.2

    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(auto_close=False)
    plotter.screenshot(str(output))
    plotter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "res" / "hopfions_h6.png",
        help="output PNG path (default: res/hopfions_h6.png)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args().output)
