#!/usr/bin/env python
# verify_msd_consistency.py - 暴力法 vs 解析法 MSD 一致性验证

import numpy as np
import time
from tqdm import tqdm

# ================= 配置区 =================
CONFIG = {
    'shape': (250, 180, 200),
    'spacing': (1.0, 1.0, 1.0),
    'n_samples': 20,  # 验证样本数（50次已足够统计稳定性）
    'random_seed': 42  # 固定种子保证可复现
}


# ================= 1. 坐标生成与统计量预计算 =================
def prepare_coords_and_stats(shape, spacing):
    """生成全量坐标并预计算 μ_p, Σ_p"""
    coords = [np.arange(s) * sp for s, sp in zip(shape, spacing)]
    Z, Y, X = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
    P = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=1).astype(np.float64)  # (N, 3)

    N = len(P)
    mu_p = P.mean(axis=0)  # (3,)

    # 🔑 总体协方差矩阵: Σ_p = E[ppᵀ] - μ_pμ_pᵀ
    # 注意: 使用 N 而非 N-1，保证与数学期望严格一致
    Sigma_p = (P.T @ P) / N - np.outer(mu_p, mu_p)  # (3, 3)
    return P, mu_p, Sigma_p


# ================= 2. 旋转矩阵构建 (Z→X→Y 固定轴) =================
def get_rot_matrix(r_deg):
    rz, rx, ry = np.deg2rad(r_deg)
    cz, sz = np.cos(rz), np.sin(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx @ Rz


# ================= 3. 暴力法 MSD =================
def msd_brute(P, t1, r1, t2, r2):
    """遍历全量体素计算 MSD (O(N))"""
    R1, R2 = get_rot_matrix(r1), get_rot_matrix(r2)
    # 向量化计算: p1 = R1·Pᵀ + t1, p2 = R2·Pᵀ + t2
    p1 = R1 @ P.T + t1[:, None]  # (3, N)
    p2 = R2 @ P.T + t2[:, None]  # (3, N)
    return np.mean(np.sum((p1 - p2) ** 2, axis=0))


# ================= 4. 解析法 MSD =================
def msd_analytical(mu_p, Sigma_p, t1, r1, t2, r2):
    """闭式公式计算 MSD (O(1))"""
    R1, R2 = get_rot_matrix(r1), get_rot_matrix(r2)
    dR = R1 - R2
    dt = t1 - t2

    term1 = np.trace(dR.T @ dR @ Sigma_p)  # tr(ΔRᵀΔR · Σ_p)
    term2 = np.sum((dR @ mu_p + dt) ** 2)  # ‖ΔR·μ_p + Δt‖²
    return term1 + term2


# ================= 5. 验证主流程 =================
def run_verification(cfg):
    print(f"🔄 初始化坐标网格与统计量 (shape={cfg['shape']})...")
    t0 = time.time()
    P, mu_p, Sigma_p = prepare_coords_and_stats(cfg['shape'], cfg['spacing'])
    print(f"✅ 预计算完成 | 体素数 N={len(P):,} | 耗时: {time.time() - t0:.2f}s\n")

    np.random.seed(cfg['random_seed'])
    abs_errors = []
    rel_errors = []

    print(f"🔍 开始验证 {cfg['n_samples']} 组随机位姿...")
    for i in tqdm(range(cfg['n_samples']), desc="验证进度"):
        # 随机位姿参数
        t1 = np.random.uniform(-25, 25, 3)
        r1 = np.random.uniform(-10, 10, 3)
        t2 = np.random.uniform(-25, 25, 3)
        r2 = np.random.uniform(-10, 10, 3)

        # 双方法计算
        v_brute = msd_brute(P, t1, r1, t2, r2)
        v_ana = msd_analytical(mu_p, Sigma_p, t1, r1, t2, r2)

        # 误差统计
        abs_err = abs(v_brute - v_ana)
        rel_err = abs_err / (v_brute + 1e-15) * 100
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

        # 打印前 3 个样本明细
        if i < 3:
            print(f"  📄 样本{i + 1}: 暴力={v_brute:.12f} | 解析={v_ana:.12f} | Δ={abs_err:.2e} | δ={rel_err:.2e}%")

    # ================= 结果汇总 =================
    abs_errors, rel_errors = np.array(abs_errors), np.array(rel_errors)
    print(f"\n{'=' * 65}")
    print(f"📊 一致性验证结果 (N={cfg['n_samples']}):")
    print(f"   • 最大绝对误差: {abs_errors.max():.2e} mm²")
    print(f"   • 平均绝对误差: {abs_errors.mean():.2e} mm²")
    print(f"   • 最大相对误差: {rel_errors.max():.2e} %")
    print(f"   • 平均相对误差: {rel_errors.mean():.2e} %")

    # 判定
    if rel_errors.max() < 1e-10:
        print(f"✅ 结论: 两种方法在 float64 机器精度范围内完全一致！")
    elif rel_errors.max() < 1e-7:
        print(f"⚠️ 结论: 存在极微小浮点累积差异（<1e-7%），数学上等价。")
    else:
        print(f"❌ 结论: 误差超出预期，请检查旋转顺序或协方差定义。")
    print(f"{'=' * 65}")

    return abs_errors, rel_errors


# ================= 6. 执行入口 =================
if __name__ == "__main__":
    abs_err, rel_err = run_verification(CONFIG)