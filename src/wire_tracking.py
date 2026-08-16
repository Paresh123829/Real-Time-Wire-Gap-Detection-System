import numpy as np
import config

MAX_JUMP_ALPHA = config.MAX_JUMP_ALPHA
MIN_JUMP_PX = config.MIN_JUMP_PX

def track_wires_across_strips(strip_detections, reference_pitch=15.0, max_jump_alpha=MAX_JUMP_ALPHA):
    """
    Monotonic Ordered Cross-Strip Wire Tracking:
    - Enforces strict candidate monotonicity per strip (W1 < W2 < ... < Wk).
    - Dynamic pitch-adaptive jump limit: max_jump_px = max(MIN_JUMP_PX, int(round(max_jump_alpha * reference_pitch))).
    - Prevents candidate identity swapping, crossings, or duplicate wire tracks.
    """
    num_strips = len(strip_detections)
    if num_strips == 0:
        return {'trajectories': [], 'total_strips': 0}

    # Dynamic pitch-relative jump distance with safety floor
    max_jump_px = max(MIN_JUMP_PX, int(round(max_jump_alpha * reference_pitch)))

    # Select primary seed strip using quality score
    strip_scores = [s.get('quality_score', 0.0) for s in strip_detections]
    seed_idx = int(np.argmax(strip_scores)) if len(strip_scores) > 0 else 0
    seed_det = strip_detections[seed_idx]
    seed_positions = seed_det['positions']
    
    if len(seed_positions) == 0:
        for s_det in strip_detections:
            if len(s_det['positions']) > 0:
                seed_positions = s_det['positions']
                break

    trajectories = []
    
    for wire_idx, init_pos in enumerate(seed_positions):
        traj = {'wire_id': wire_idx + 1, 'strip_positions': {seed_idx: float(init_pos)}}
        curr_pos = float(init_pos)

        # Track forward
        for s in range(seed_idx + 1, num_strips):
            cands = strip_detections[s]['positions']
            if len(cands) > 0:
                valid_cands = [c for c in cands if abs(c - curr_pos) <= max_jump_px]
                if valid_cands:
                    best_c = min(valid_cands, key=lambda c: abs(c - curr_pos))
                    traj['strip_positions'][s] = float(best_c)
                    curr_pos = float(best_c)

        # Track backward
        curr_pos = float(init_pos)
        for s in range(seed_idx - 1, -1, -1):
            cands = strip_detections[s]['positions']
            if len(cands) > 0:
                valid_cands = [c for c in cands if abs(c - curr_pos) <= max_jump_px]
                if valid_cands:
                    best_c = min(valid_cands, key=lambda c: abs(c - curr_pos))
                    traj['strip_positions'][s] = float(best_c)
                    curr_pos = float(best_c)

        traj['support_count'] = len(traj['strip_positions'])
        traj['support_ratio'] = float(traj['support_count']) / float(num_strips)
        trajectories.append(traj)

    # Enforce Monotonicity & Ordered Duplicate Resolution (W1 < W2 < ... < Wk)
    monotonic_trajectories = []
    prev_positions = {s: -1.0 for s in range(num_strips)}

    for traj in sorted(trajectories, key=lambda t: np.mean(list(t['strip_positions'].values()))):
        is_valid = True
        filtered_strip_pos = {}
        for s, pos in sorted(traj['strip_positions'].items()):
            if pos > prev_positions[s]:
                filtered_strip_pos[s] = pos
            else:
                is_valid = False

        if len(filtered_strip_pos) > 0:
            for s, pos in filtered_strip_pos.items():
                prev_positions[s] = pos
            traj['strip_positions'] = filtered_strip_pos
            traj['support_count'] = len(filtered_strip_pos)
            traj['support_ratio'] = float(len(filtered_strip_pos)) / float(num_strips)
            monotonic_trajectories.append(traj)

    return {
        'trajectories': monotonic_trajectories,
        'total_strips': num_strips,
        'seed_strip': seed_idx,
        'max_jump_px': max_jump_px
    }

def compute_robust_wire_positions(trajectories):
    """
    Computes robust wire positions across trajectories using median strip position.
    Also computes overall tracking support score across all trajectories.
    """
    positions = []
    support_scores = []
    
    for traj in trajectories:
        strip_vals = list(traj['strip_positions'].values())
        if strip_vals:
            pos_med = float(np.median(strip_vals))
            positions.append(pos_med)
            support_scores.append(traj['support_ratio'])

    positions = np.array(sorted(positions), dtype=np.float32)
    mean_support = float(np.mean(support_scores)) if support_scores else 0.0

    return {
        'positions': positions,
        'support_scores': support_scores,
        'mean_support_score': mean_support
    }
