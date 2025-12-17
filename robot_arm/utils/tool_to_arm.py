import numpy as np

def get_tool2arm_rt(alpha, beta, param_cz):
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    c, z = param_cz
    sc, cc = np.sin(c), np.cos(c)

    R_alpha = np.array([
        [1, 0, 0],
        [0, ca, sa],
        [0, -sa, ca]
    ])

    R_beta = np.array([
        [cb, 0, -sb],
        [0, 1, 0],
        [sb, 0, cb]
    ])

    R_c = np.array([
        [cc, sc, 0],
        [-sc, cc, 0],
        [0, 0, 1]
    ])
    R = R_alpha @ R_c @ R_beta

    # R = np.array([
    #     [cb * cc, sc, -sb * cc],
    #     [-cb * sc * ca + sb * sa, cc * ca, cb * sa + sb * sc * ca],
    #     [sb * ca + cb * sc * sa, -cc * sa, cb * ca - sb * sc * sa]
    # ])

    T = np.array([
        [0],
        [z * sa],
        [z * ca]
    ])

    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3:4] = T

    return R, T

