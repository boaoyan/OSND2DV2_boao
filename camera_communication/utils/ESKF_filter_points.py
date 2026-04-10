import numpy as np


def pAB2pC_angles(pAB):
    """
    Convert [pA; pB] (6,) to [pC; phi; theta] (5,)
    pC = (pA + pB) / 2
    nV = (pB - pA) / ||pB - pA||  (unit vector)
    phi = atan2(nVy, nVx)
    theta = asin(nVz)
    """
    pA = pAB[:3]
    pB = pAB[3:]
    pC = (pA + pB) / 2.0
    nV = pB - pA
    norm_nV = np.linalg.norm(nV)
    if norm_nV < 1e-8:
        # Degenerate case: points coincide
        nV_unit = np.array([1.0, 0.0, 0.0])
    else:
        nV_unit = nV / norm_nV
    phi = np.arctan2(nV_unit[1], nV_unit[0])
    theta = np.arcsin(np.clip(nV_unit[2], -1.0, 1.0))  # avoid numerical error
    pC_angles = np.concatenate([pC, [phi, theta]])
    return pC_angles, nV_unit


def pC_angles2pAB(pC_angles, d):
    """
    Convert [pC; phi; theta] (5,) back to [pA; pB] (6,)
    nV = [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)]
    pA = pC - (d/2) * nV
    pB = pC + (d/2) * nV
    """
    pC = pC_angles[:3]
    phi = pC_angles[3]
    theta = pC_angles[4]
    cos_theta = np.cos(theta)
    nV = np.array([
        np.cos(phi) * cos_theta,
        np.sin(phi) * cos_theta,
        np.sin(theta)
    ])
    pA = pC - (d / 2.0) * nV
    pB = pC + (d / 2.0) * nV
    pAB = np.concatenate([pA, pB])
    return pAB, nV


def ESKF_1step_2Points(state, P, measures, d, sigma_Q, sigma_R):
    """
    One step of ESKF for two rigid points with fixed distance d.

    Parameters:
    ----------
    state : (6,) array
        Current estimate of [pA; pB]
    P : (5,5) array
        Covariance of internal 5D state [pC; phi; theta]
    measures : (6,) array
        Measured [pA_meas; pB_meas]
    d : float
        Fixed distance between pA and pB
    sigma_Q : array-like, [sigma_trans, sigma_rot]
        Process noise std per step
    sigma_R : float
        Measurement noise std (isotropic in Cartesian space)

    Returns:
    -------
    state_filted : (6,) array
        Updated [pA; pB]
    P_filted : (5,5) array
        Updated covariance
    """
    # --- Process noise covariance Q (5x5 diagonal) ---
    sigma_trans, sigma_rot = sigma_Q
    Q_diag = np.array([sigma_trans ** 2] * 3 + [sigma_rot ** 2] * 2)
    Q = np.diag(Q_diag)

    # --- Prediction ---
    P_pred = P + Q  # (5,5)

    # --- Convert to internal 5D representation ---
    pC_angles_meas, _ = pAB2pC_angles(measures)
    pC_angles_pred, nV = pAB2pC_angles(state)
    y = pC_angles_meas - pC_angles_pred  # (5,)

    # --- Compute R5 (5x5 diagonal) based on current orientation ---
    nvz = nV[2]
    # Avoid division by zero near poles (|nvz| -> 1)
    eps = 1e-6
    denom = max(1.0 - nvz ** 2, eps)

    var_pC = (sigma_R ** 2) / 2.0  # variance for each pC coordinate
    var_phi = (2.0 * sigma_R ** 2) / (d ** 2 * denom)
    var_theta = (2.0 * sigma_R ** 2 * (1.0 + nvz ** 2 - nvz ** 4)) / (d ** 2 * denom)

    R5_diag = np.array([var_pC, var_pC, var_pC, var_phi, var_theta])
    R5 = np.diag(R5_diag)

    # --- Kalman gain (element-wise since P_pred and R5 are diagonal) ---
    # Use Joseph form for numerical stability
    P_pred_diag = np.diag(P_pred)
    K_diag = P_pred_diag / (P_pred_diag + R5_diag)
    K = np.diag(K_diag)  # (5,5)

    # --- Update covariance (Joseph form) ---
    I5 = np.eye(5)
    P_updated = (I5 - K) @ P_pred @ (I5 - K).T + K @ R5 @ K.T

    # --- Update state ---
    delta_state = K @ y  # (5,)
    pC_angles_updated = pC_angles_pred + delta_state
    state_filted, _ = pC_angles2pAB(pC_angles_updated, d)

    return state_filted, P_updated