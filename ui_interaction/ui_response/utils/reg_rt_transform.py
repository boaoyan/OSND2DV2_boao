import numpy as np

from ui_interaction.ui_response.utils.euler_rotation_transform import rotation_to_euler_angles, \
    euler_angles_to_rotation, reg_rt_update_per_dof


def reg_rt_transform(rt_ct2o_sz_norm, rt_ct2o_sc_norm, rt_ct2o_sz, rt_ct2o_sc):
    # rt_osz2osc = rt_ct2o_sc_norm @ np.linalg.inv(rt_ct2o_sz_norm)
    rt_osc2osz = rt_ct2o_sz_norm @ np.linalg.inv(rt_ct2o_sc_norm)

    rt_ct2o_sz_tran = rt_osc2osz @ rt_ct2o_sc

    euler_ct2o_sz_norm = rotation_to_euler_angles(rt_ct2o_sz_norm)
    euler_ct2o_sz = rotation_to_euler_angles(rt_ct2o_sz)
    euler_ct2o_sz_tran = rotation_to_euler_angles(rt_ct2o_sz_tran)
    euler_dev = euler_ct2o_sz - euler_ct2o_sz_norm
    euler_dev_tran = euler_ct2o_sz_tran - euler_ct2o_sz_norm

    return euler_dev, euler_dev_tran

def reg_rt_update(rt_ct2o_sz, rt_ct2o_sc, reg_var, reg_var_tran):
    rt_ct2o_sz_norm = [[-1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 500],
                       [0, 0, 0, 1]]
    rt_ct2o_sc_norm = [[0, -1, 0, 0],
                       [0, 0, 1, 0],
                       [-1, 0, 0, 500],
                       [0, 0, 0, 1]]

    rt_osz2osc = rt_ct2o_sc_norm @ np.linalg.inv(rt_ct2o_sz_norm)
    rt_osc2osz = rt_ct2o_sz_norm @ np.linalg.inv(rt_ct2o_sc_norm)

    rt_ct2o_sz_tran = rt_osc2osz @ rt_ct2o_sc


    euler_ct2o_sz = rotation_to_euler_angles(rt_ct2o_sz)
    # rt_ct2o_sz_new = euler_angles_to_rotation(euler_ct2o_sz)
    euler_ct2o_sz_tran = rotation_to_euler_angles(rt_ct2o_sz_tran)


    euler_ct2o_sz_update = reg_rt_update_per_dof(
                                    euler_ct2o_sz,
                                    euler_ct2o_sz_tran,
                                    reg_var,
                                    reg_var_tran
                                )


    # euler_ct2o_sz_update = (reg_var_tran**2/(reg_var**2 + reg_var_tran**2)) * euler_ct2o_sz + (reg_var**2/(reg_var**2 + reg_var_tran**2)) * euler_ct2o_sz_tran

    rt_ct2o_sz_update = euler_angles_to_rotation(euler_ct2o_sz_update)
    rt_ct2o_sc_update = rt_osz2osc @ rt_ct2o_sz_update

    return rt_ct2o_sz_update,rt_ct2o_sc_update



