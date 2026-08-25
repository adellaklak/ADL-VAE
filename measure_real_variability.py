import numpy as np

KINEMATIC_TREE = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6),
    (20, 8), (8, 9), (9, 10), (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]
LEG_BONES = [(0,12),(12,13),(13,14),(14,15)]  # jambe gauche complete, pour taille corps

def joint_angle(seq, ja, jc, jb):
    v1 = seq[:, ja, :] - seq[:, jc, :]
    v2 = seq[:, jb, :] - seq[:, jc, :]
    cos_a = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def instance_stats(sv, up_axis=1):
    """sv: (T,25,3) deja centre UNE FOIS (frame 0), pas par frame."""
    T = sv.shape[0]
    duration = T  # a 30fps -> duree en secondes = T/30

    # taille corps: somme jambe + tronc (bornes stables, peu affectees par la pose)
    body_size = sum(np.linalg.norm(sv[:, a, :] - sv[:, b, :], axis=-1).mean() for a, b in LEG_BONES)
    body_size += np.linalg.norm(sv[:, 1, :] - sv[:, 20, :], axis=-1).mean()  # SpineMid->SpineShoulder

    spinebase_vrange = sv[:, 0, up_axis].max() - sv[:, 0, up_axis].min()
    head_vrange = sv[:, 3, up_axis].max() - sv[:, 3, up_axis].min()
    knee_l = joint_angle(sv, 12, 13, 14); knee_r = joint_angle(sv, 16, 17, 18)
    knee_min = min(knee_l.min(), knee_r.min())

    return {'duration_frames': duration, 'body_size': body_size,
            'spinebase_vrange': spinebase_vrange, 'head_vrange': head_vrange,
            'knee_min': knee_min}

d = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/Neuron/data/ntu60/NTU60_CS.npz')
x_test = d['x_test'].reshape(-1, 300, 2, 25, 3)
y_test = d['y_test'].argmax(axis=1)
body1_all = x_test[:, :, 0, :, :]

CLASSES = [3, 5, 9, 10, 11, 12, 15, 19, 26, 40, 42, 47]
NAMES = {3:"brush hair",5:"pick up",9:"clapping",10:"reading",11:"writing",12:"tear up paper",
         15:"put on a shoe",19:"put on hat",26:"jump up",40:"sneeze/cough",42:"falling down",47:"nausea"}

FEATS = ['duration_frames', 'body_size', 'spinebase_vrange', 'head_vrange', 'knee_min']
print(f"{'classe':<14} {'feature':<18} {'moy':>8} {'std':>8} {'min':>8} {'max':>8} {'cv%':>6}")
for c in CLASSES:
    mask = y_test == c
    seqs_raw = body1_all[mask]
    stats_list = []
    for s in seqs_raw:
        valid = np.abs(s).sum(axis=(1, 2)) > 1e-6
        sv = s[valid]
        if sv.shape[0] < 5: continue
        sv = sv - sv[0, 0, :]
        stats_list.append(instance_stats(sv))
    if not stats_list: continue
    for f in FEATS:
        vals = np.array([st[f] for st in stats_list])
        cv = 100 * vals.std() / (abs(vals.mean()) + 1e-8)
        print(f"{NAMES[c]:<14} {f:<18} {vals.mean():>8.2f} {vals.std():>8.2f} {vals.min():>8.2f} {vals.max():>8.2f} {cv:>5.1f}%")
    print()
