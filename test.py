import numpy as np

# p1 = np.array([-30.9976551, -38.72113799, 28.7174386])
# p2 = np.array([32.17165348,  39.0495092, -27.03646642])
p1 = np.array([37.94320527,  5.06780067, 44.40407641])
p2 = np.array([-36.66520007,  -6.29309517, -42.13093802])
distance = np.linalg.norm(p2 - p1)
print(f"两点间距离: {distance:.4f}")