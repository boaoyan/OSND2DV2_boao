import numpy as np

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
