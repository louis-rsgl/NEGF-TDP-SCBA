import numpy as np

def expc(z, t, tol=1e-10):
    """
    Computes expc(z, t) = (exp(i z t) - 1) / (i z), with the removable
    limit expc(0, t) = t.

    Supports scalar or vector t.
    """
    z = np.asarray(z, dtype=np.complex128)
    t = np.asarray(t, dtype=np.float64)

    z_b, t_b = np.broadcast_arrays(z, t)
    out = np.empty_like(z_b, dtype=np.complex128)

    mask = np.abs(z_b * t_b) < tol
    out[mask] = (
        t_b[mask]
        + 0.5j * z_b[mask] * t_b[mask] ** 2
        - z_b[mask] ** 2 * t_b[mask] ** 3 / 6.0
    )
    out[~mask] = (
        np.expm1(1j * z_b[~mask] * t_b[~mask]) / (1j * z_b[~mask])
    )

    return out

def fermi_dirac(z, beta, mu=0.0):
    z = np.asarray(z, dtype=np.complex128)
    s = beta * (z - mu)
    re = s.real
    out = np.empty_like(s, dtype=np.complex128)

    m = re > 0
    t = np.exp(-s[m])
    out[m] = t / (1.0 + t)

    m2 = ~m
    t2 = np.exp(s[m2])
    out[m2] = 1.0 / (1.0 + t2)
    return out

def bose_einstein(z, beta, mu0=0.0):
    z = np.asarray(z, dtype=np.complex128)
    s = beta * (z - mu0)
    re = s.real
    out = np.empty_like(s, dtype=np.complex128)

    m = re > 0
    t = np.exp(-s[m])
    out[m] = t / (1.0 - t)

    m2 = ~m
    t2 = np.exp(s[m2])
    out[m2] = 1.0 / (t2 - 1.0)
    return out
