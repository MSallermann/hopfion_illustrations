import numpy as np


def compute_linking_number(curve1, curve2, num_points=1000, t_range=(0, 2 * np.pi)):
    """
    Compute the linking number of two closed curves using the Gauss linking integral.

    Lk = 1/(4π) ∮_C1 ∮_C2 ((r1 - r2) · (dr1 × dr2)) / ||r1 - r2||^3 ds dt.

    Parameters:
      curve1, curve2 : callable
          Functions mapping a scalar t to a numpy array of shape (3,).
      num_points : int, optional
          Number of discretization points (default 1000).
      t_range : tuple, optional
          Parameter interval (default (0, 2π)).

    Returns:
      float
          The computed linking number.
    """
    t_min, t_max = t_range
    t1 = np.linspace(t_min, t_max, num_points, endpoint=False)
    t2 = np.linspace(t_min, t_max, num_points, endpoint=False)

    r1 = np.array([curve1(t) for t in t1])
    r2 = np.array([curve2(t) for t in t2])

    dt = (t_max - t_min) / num_points
    dr1 = np.gradient(r1, dt, axis=0, edge_order=2)
    dr2 = np.gradient(r2, dt, axis=0, edge_order=2)

    diff = r1[:, None, :] - r2[None, :, :]
    norm_diff = np.linalg.norm(diff, axis=2)
    cross_prod = np.cross(dr1[:, None, :], dr2[None, :, :])
    dot_prod = np.sum(diff * cross_prod, axis=2)
    integrand = dot_prod / (norm_diff**3)

    linking = np.sum(integrand) * (dt**2) / (4 * np.pi)
    return linking


def stereographic_projection_south(x):
    """
    Stereographic projection from S^3 (in R^4) to R^3, projecting from the south pole.
    For x = (x1, x2, x3, x4), returns (x1/(1+x4), x2/(1+x4), x3/(1+x4)).
    """
    return np.array([x[0] / (1 + x[3]), x[1] / (1 + x[3]), x[2] / (1 + x[3])])


def hopf_link_circle1(t):
    """
    Circle 1 on S^3 corresponding to (e^(it)/√2, e^(it)/√2).
    """
    x1 = np.cos(t) / np.sqrt(2)
    x2 = np.sin(t) / np.sqrt(2)
    x3 = np.cos(t) / np.sqrt(2)
    x4 = np.sin(t) / np.sqrt(2)
    return stereographic_projection_south(np.array([x1, x2, x3, x4]))


def hopf_link_circle2(t):
    """
    Circle 2 on S^3 corresponding to (e^(it)/√2, -e^(it)/√2).
    """
    x1 = np.cos(t) / np.sqrt(2)
    x2 = np.sin(t) / np.sqrt(2)
    x3 = -np.cos(t) / np.sqrt(2)
    x4 = -np.sin(t) / np.sqrt(2)
    return stereographic_projection_south(np.array([x1, x2, x3, x4]))


if __name__ == "__main__":
    lk = compute_linking_number(hopf_link_circle1, hopf_link_circle2, num_points=1000)
    print("Linking Number for the Hopf link:", lk)
