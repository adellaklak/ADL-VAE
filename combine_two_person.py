import numpy as np

# offsets mesures sur donnees reelles (measure_relative_position.py)
STATIC_OFFSETS = {
    51: np.array([-0.012, -0.009, 0.005]),   # pushing
    56: np.array([0.039, -0.055, -0.108]),   # touch pocket
}
# walking towards/apart: distance tres variable (std 0.48-0.58) -> trajectoire, pas offset fixe
DYNAMIC_CLASSES = {
    58: {'start_dist': 2.2, 'end_dist': 0.9, 'axis': 0},   # walking towards: se rapprochent
    59: {'start_dist': 0.9, 'end_dist': 2.2, 'axis': 0},   # walking apart: s'eloignent
}

def align_length(seqA, seqB):
    """Aligne 2 sequences a la meme longueur par resampling (pas troncature -- garde tout le mouvement)."""
    T = max(seqA.shape[0], seqB.shape[0])
    def resample(seq, T_target):
        T_src = seq.shape[0]
        if T_src == T_target: return seq
        idx = np.linspace(0, T_src - 1, T_target)
        idx_floor = np.floor(idx).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, T_src - 1)
        frac = (idx - idx_floor)[:, None, None]
        return seq[idx_floor] * (1 - frac) + seq[idx_ceil] * frac
    return resample(seqA, T).astype(np.float32), resample(seqB, T).astype(np.float32)

def combine_static(seqA, seqB, cls):
    """seqA, seqB: (T,25,3) deja retargetees/FK/mains dupliquees, chacune centree sur sa propre racine."""
    seqA, seqB = align_length(seqA, seqB)
    offset = STATIC_OFFSETS[cls]
    seqB_shifted = seqB + offset  # B se retrouve a la bonne distance/direction de A
    return seqA, seqB_shifted

def combine_dynamic(seqA, seqB, cls):
    """walking towards/apart: distance evolue lineairement dans le temps."""
    seqA, seqB = align_length(seqA, seqB)
    T = seqA.shape[0]
    params = DYNAMIC_CLASSES[cls]
    dist_traj = np.linspace(params['start_dist'], params['end_dist'], T)
    offset = np.zeros((T, 3), dtype=np.float32)
    offset[:, params['axis']] = dist_traj
    seqB_shifted = seqB + offset[:, None, :]
    return seqA, seqB_shifted

def combine_two_person(seqA, seqB, cls):
    if cls in STATIC_OFFSETS:
        return combine_static(seqA, seqB, cls)
    elif cls in DYNAMIC_CLASSES:
        return combine_dynamic(seqA, seqB, cls)
    else:
        raise ValueError(f"pas de config pour classe {cls}")

def to_shiftgcn_input_2person(seqA, seqB, max_T=300):
    T = min(seqA.shape[0], max_T)
    x = np.zeros((1, 3, max_T, 25, 2), dtype=np.float32)
    x[0, :, :T, :, 0] = seqA[:T].transpose(2, 0, 1)
    x[0, :, :T, :, 1] = seqB[:T].transpose(2, 0, 1)
    return x
