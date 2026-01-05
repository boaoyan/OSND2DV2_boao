import numpy as np

sid = 500 #光源到投影中心的距离

d_s2p = 800 #光源到投影面的距离
im_sz = [512, 512]
pan_sz = [210, 210]
a = np.array([[-d_s2p * im_sz[0] / pan_sz[0], 0, (im_sz[0]) / 2],
                     [0, -d_s2p * im_sz[1] / pan_sz[1], (im_sz[1]) / 2],
                     [0, 0, 1]])

np.save('data/trans_matrix/spine107/a_arm_new.npy', a)