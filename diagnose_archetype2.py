import numpy as np

NTU_FROM_H3D = {
    0: 0, 1: 6, 2: 12, 3: 15, 4: 16, 5: 18, 6: 20, 7: 20,
    8: 17, 9: 19, 10: 21, 11: 21, 12: 1, 13: 4, 14: 7, 15: 10,
    16: 2, 17: 5, 18: 8, 19: 11, 20: 9, 21: 20, 22: 20, 23: 21, 24: 21,
}
KINEMATIC_TREE = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6),
    (20, 8), (8, 9), (9, 10), (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

def resample_20_to_30(seq20):
    T = seq20.shape[0]
    new_T = int(round(T * 30 / 20))
    src_idx = np.linspace(0, T - 1, new_T)
    idx_floor = np.floor(src_idx).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, T - 1)
    frac = (src_idx - idx_floor)[:, None, None]
    return (seq20[idx_floor] * (1 - frac) + seq20[idx_ceil] * frac).astype(np.float32)

def retarget_naive_global(seq22):
    """Centrage UNE FOIS sur la racine de la frame 0, pas par frame -- preserve la trajectoire."""
    T = seq22.shape[0]
    seq25 = np.zeros((T, 25, 3), dtype=np.float32)
    for ntu_j, h3d_j in NTU_FROM_H3D.items():
        seq25[:, ntu_j, :] = seq22[:, h3d_j, :]
    seq25 = seq25 - seq25[0, 0, :]
    return seq25

def retarget_fk_scale(naive_seq25, edge_length):
    T = naive_seq25.shape[0]
    corrected = np.zeros_like(naive_seq25)
    corrected[:, 0, :] = naive_seq25[:, 0, :]
    for a, b in KINEMATIC_TREE:
        direction = naive_seq25[:, b, :] - naive_seq25[:, a, :]
        direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
        corrected[:, b, :] = corrected[:, a, :] + direction * edge_length[(a, b)]
    return corrected

def joint_angle(seq, ja, jc, jb):
    v1 = seq[:, ja, :] - seq[:, jc, :]
    v2 = seq[:, jb, :] - seq[:, jc, :]
    cos_a = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def archetype_features_global(seq25_global, up_axis=1):
    """seq25_global: centre UNE FOIS (frame 0), pas par frame."""
    knee_l = joint_angle(seq25_global, 12, 13, 14)
    knee_r = joint_angle(seq25_global, 16, 17, 18)
    knee_min = min(knee_l.min(), knee_r.min())
    torso = seq25_global[:, 20, :] - seq25_global[:, 0, :]
    up = np.zeros(3); up[up_axis] = 1
    cos_a = (torso * up).sum(-1) / (np.linalg.norm(torso, axis=-1) + 1e-8)
    hip_flex_max = np.degrees(np.arccos(np.clip(cos_a, -1, 1))).max()
    spinebase_vrange_global = seq25_global[:, 0, up_axis].max() - seq25_global[:, 0, up_axis].min()
    head_vrange_global = seq25_global[:, 3, up_axis].max() - seq25_global[:, 3, up_axis].min()
    head_rel = seq25_global[:, 3, up_axis] - seq25_global[:, 0, up_axis]
    head_vrange_relative = head_rel.max() - head_rel.min()
    return {'knee_min': knee_min, 'hip_flex_max': hip_flex_max,
            'spinebase_vrange_global': spinebase_vrange_global,
            'head_vrange_global': head_vrange_global,
            'head_vrange_relative': head_vrange_relative}

# ---------- verif signal brut AVANT tout retargeting ----------
RESULTS_DIR_1 = '/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/MotionGPT3/results/motgpt/debug--MoT_vae_trainemb_cls1_cfgrand_2optim_lr2_lr1/samples_2026-08-08-03-41-21'
seq_falling_raw = np.load(f'{RESULTS_DIR_1}/28_out.npy')
if seq_falling_raw.ndim == 4: seq_falling_raw = seq_falling_raw.squeeze(0)
head_h3d, foot_h3d = 15, 10
print("Verif signal vertical BRUT (avant retargeting), falling down genere:")
print(f"  Head (joint 15) axe1 range brut: {seq_falling_raw[:,head_h3d,1].max()-seq_falling_raw[:,head_h3d,1].min():.3f}")
print(f"  Pelvis (joint 0) axe1 range brut: {seq_falling_raw[:,0,1].max()-seq_falling_raw[:,0,1].min():.3f}\n")

# ---------- donnees reelles ----------
d = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/Neuron/data/ntu60/NTU60_CS.npz')
x_test = d['x_test'].reshape(-1, 300, 2, 25, 3)
y_test = d['y_test'].argmax(axis=1)
body1_all = x_test[:, :, 0, :, :]
valid_all = np.abs(body1_all).sum(axis=(2, 3)) > 1e-6

edge_length = {}
for a, b in KINEMATIC_TREE:
    dist = np.linalg.norm(body1_all[:, :, a, :] - body1_all[:, :, b, :], axis=-1)
    edge_length[(a, b)] = dist[valid_all].mean()

def real_archetype_avg_global(c, max_seqs=40, seed=5):
    mask = y_test == c
    seqs_raw = body1_all[mask][:max_seqs]
    feats_list = []
    for s in seqs_raw:
        valid = np.abs(s).sum(axis=(1, 2)) > 1e-6
        sv = s[valid]
        if sv.shape[0] < 3: continue
        sv_global = sv - sv[0, 0, :]
        feats_list.append(archetype_features_global(sv_global))
    return {k: np.mean([f[k] for f in feats_list]) for k in feats_list[0]}

BEST_VARIANT = {
    5:  (RESULTS_DIR_1, 3, 15),
    42: (RESULTS_DIR_1, 28, 58),
    47: (RESULTS_DIR_1, 31, 3),
}
NAMES = {5:"pick up", 15:"put on a shoe", 42:"falling down", 58:"walking towards",
         47:"nausea/vomiting", 3:"brush hair"}

FEATS = ['knee_min', 'hip_flex_max', 'spinebase_vrange_global', 'head_vrange_global', 'head_vrange_relative']
print(f"{'classe':<16} {'source':<14} " + " ".join(f"{f:>22}" for f in FEATS))
for c, (rdir, idx, confusor) in BEST_VARIANT.items():
    seq22 = np.load(f"{rdir}/{idx}_out.npy")
    if seq22.ndim == 4: seq22 = seq22.squeeze(0)
    seq30 = resample_20_to_30(seq22)
    naive25 = retarget_naive_global(seq30)
    scaled25 = retarget_fk_scale(naive25, edge_length)
    gen_feat = archetype_features_global(scaled25)
    real_true = real_archetype_avg_global(c)

    print(f"{NAMES[c]:<16} {'GENERE':<14} " + " ".join(f"{gen_feat[f]:>22.2f}" for f in FEATS))
    print(f"{'':<16} {'reel (vrai)':<14} " + " ".join(f"{real_true[f]:>22.2f}" for f in FEATS))
    if confusor:
        real_conf = real_archetype_avg_global(confusor)
        print(f"{'':<16} {'reel ('+NAMES[confusor]+')':<14} " + " ".join(f"{real_conf[f]:>22.2f}" for f in FEATS))
    print()
