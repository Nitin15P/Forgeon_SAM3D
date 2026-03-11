"""
Inverse Dynamics using nimblephysics native EOM.

GRF estimation: static equilibrium assumption distributes body-weight
between feet in contact, proportional to their proximity to the ground.
Joint torques computed via DART skeleton.computeInverseDynamics().

Usage:
    python run_inverse_dynamics.py \
        --osim  <path/to/model.osim> \
        --mot   <path/to/ik.mot> \
        --out   <output_folder>
"""

import argparse
import os
import numpy as np
import nimblephysics as nimble


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_mot(mot_path):
    times, rows, dof_names = [], [], []
    in_data = False
    with open(mot_path) as f:
        for line in f:
            s = line.rstrip()
            if s.lower() == 'endheader':
                in_data = True
            elif in_data and not dof_names:
                dof_names = s.split('\t')
            elif in_data and dof_names:
                vals = list(map(float, s.split('\t')))
                times.append(vals[0])
                rows.append(vals[1:])
    poses = np.array(rows, dtype=np.float64).T  # (ndof, nframes)
    times = np.array(times, dtype=np.float64)
    return times, dof_names[1:], poses


def finite_diff(x, dt):
    n = x.shape[1]
    v = np.zeros_like(x)
    a = np.zeros_like(x)
    v[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2 * dt)
    v[:, 0] = v[:, 1]; v[:, -1] = v[:, -2]
    a[:, 1:-1] = (x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]) / (dt**2)
    a[:, 0] = a[:, 1]; a[:, -1] = a[:, -2]
    return v, a


def write_sto(path, times, col_names, data):
    with open(path, 'w') as f:
        f.write('Inverse Dynamics\n')
        f.write(f'nRows={len(times)}\n')
        f.write(f'nColumns={data.shape[1]+1}\n')
        f.write('inDegrees=no\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(col_names) + '\n')
        for i, t in enumerate(times):
            f.write(f'{t:.6f}\t' + '\t'.join(f'{v:.8f}' for v in data[i]) + '\n')
    print(f'Wrote {path}')


# ─── GRF estimation ───────────────────────────────────────────────────────────

def estimate_grf(skel, foot_bodies, mass_kg, ground_y, contact_threshold=0.04):
    """
    Estimate GRF for the current skeleton pose using static equilibrium.

    - Foot bodies within `contact_threshold` metres of `ground_y` are in contact.
    - Vertical GRF = body weight, distributed across feet inversely proportional
      to height above ground (closer foot takes more load).
    - Horizontal GRF = 0 (no friction data available).

    Returns:
        dict: {body_name: np.ndarray(6,)} — spatial wrench [tx,ty,tz,fx,fy,fz]
              in world frame, applied at the foot's world position.
              Y-down convention: vertical force is in +Y direction (upward = -Y).
    """
    g = abs(skel.getGravity()[1]) if abs(skel.getGravity()[1]) > 0.1 else 9.81
    body_weight = mass_kg * g   # N

    # Foot heights above ground (in Y-down: ground at y=0, foot above = negative y)
    foot_info = {}
    for body in foot_bodies:
        world_pos = body.getWorldTransform().translation()
        # Height above ground: in Y-down system, foot is "above" ground when y < 0
        # (ground is at y=0, feet at y≤0). Height = -y (positive = above ground).
        height_above_ground = -world_pos[1]   # positive when above ground
        foot_info[body.getName()] = {
            'body': body,
            'world_pos': world_pos.copy(),
            'height': height_above_ground,
        }

    # Determine which feet are in contact
    contact_feet = {name: info for name, info in foot_info.items()
                    if info['height'] < contact_threshold}

    grf_result = {name: np.zeros(6) for name in [b.getName() for b in foot_bodies]}

    # If no foot is within threshold, use both feet with full gravity compensation
    if not contact_feet:
        contact_feet = foot_info

    # Distribute weight inversely proportional to height above ground.
    # Foot AT ground (height≈0) gets most load.
    eps = 1e-4
    weights = {name: 1.0 / (info['height'] + eps)
               for name, info in contact_feet.items()}
    total_w = sum(weights.values())

    for name, info in contact_feet.items():
        frac = weights[name] / total_w
        F_vertical = frac * body_weight   # N, upward
        # In Y-down: upward force = -Y direction (fy = -F_vertical for Y-down)
        # But ResidualForceHelper expects spatial wrench in world frame:
        #   [torque (3), force (3)] — sign convention matches world axes
        # In Y-down, "up" = -Y, so upward reaction force has fy = -F_vertical
        grf_result[name] = np.array([0, 0, 0,   # torques (Nm)
                                     0, -F_vertical, 0])  # forces (N)

    return grf_result


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--osim', required=True)
    parser.add_argument('--mot',  required=True)
    parser.add_argument('--out',  required=True)
    parser.add_argument('--mass', type=float, default=None)
    parser.add_argument('--contact-threshold', type=float, default=0.5,
                        help='Foot height threshold for ground contact (m), default 0.5')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Load skeleton
    print('Loading skeleton...')
    parsed = nimble.biomechanics.OpenSimParser.parseOsim(args.osim, ignoreGeometry=True)
    skel = parsed.skeleton
    if args.mass:
        scale = args.mass / skel.getMass()
        for i in range(skel.getNumBodyNodes()):
            skel.getBodyNode(i).setMass(skel.getBodyNode(i).getMass() * scale)
    print(f'  DOFs: {skel.getNumDofs()},  Mass: {skel.getMass():.2f} kg')

    # 2. Load IK
    print('Loading IK motion...')
    times, mot_dofs, poses_mot = load_mot(args.mot)
    nframes = poses_mot.shape[1]
    dt = float(np.median(np.diff(times)))
    print(f'  Frames: {nframes},  dt: {dt:.4f} s ({1/dt:.1f} Hz)')

    skel_dof_names = [skel.getDofByIndex(i).getName() for i in range(skel.getNumDofs())]
    poses = np.zeros((skel.getNumDofs(), nframes), dtype=np.float64)
    for i, name in enumerate(skel_dof_names):
        if name in mot_dofs:
            poses[i] = poses_mot[mot_dofs.index(name)]
    print(f'  Matched {sum(1 for n in skel_dof_names if n in mot_dofs)}/{skel.getNumDofs()} DOFs')

    # 3. Normalize: center X/Z, shift Y so ground = 0 (Y-down camera space)
    print('Normalizing coordinate system...')
    tx_i = skel_dof_names.index('pelvis_tx') if 'pelvis_tx' in skel_dof_names else 3
    ty_i = skel_dof_names.index('pelvis_ty') if 'pelvis_ty' in skel_dof_names else 4
    tz_i = skel_dof_names.index('pelvis_tz') if 'pelvis_tz' in skel_dof_names else 5
    poses[tx_i, :] -= np.mean(poses[tx_i, :])
    poses[tz_i, :] -= np.mean(poses[tz_i, :])

    max_foot_y = -np.inf
    for t in range(nframes):
        skel.setPositions(poses[:, t])
        for bname in ['calcn_r', 'calcn_l']:
            body = skel.getBodyNode(bname)
            if body:
                y = body.getWorldTransform().translation()[1]
                if y > max_foot_y:
                    max_foot_y = y
    poses[ty_i, :] -= max_foot_y
    print(f'  Ground at Y=0 (shifted by {-max_foot_y:.4f} m)')

    # 4. Finite difference
    vels, accs = finite_diff(poses, dt)

    # 5. Set gravity: in Y-down camera space, gravity = +Y
    skel.setGravity(np.array([0, 9.81, 0]))

    # 6. Identify foot contact bodies
    foot_body_names = ['calcn_r', 'toes_r', 'calcn_l', 'toes_l']
    foot_bodies = [b for b in (skel.getBodyNode(n) for n in foot_body_names) if b]
    print(f'  Contact bodies: {[b.getName() for b in foot_bodies]}')

    # 7. Compute ID frame by frame
    print('Computing inverse dynamics...')
    mass_kg = args.mass if args.mass else skel.getMass()
    n_contact_frames = 0

    torques_all   = np.zeros((nframes, skel.getNumDofs()), dtype=np.float64)
    residuals_all = np.zeros((nframes, 6), dtype=np.float64)
    grf_all       = np.zeros((nframes, len(foot_bodies) * 6), dtype=np.float64)
    contact_all   = np.zeros((nframes, len(foot_bodies)), dtype=np.float64)  # 1=contact

    for t in range(nframes):
        q   = poses[:, t]
        dq  = vels[:, t]
        ddq = accs[:, t]

        skel.setPositions(q)
        skel.setVelocities(dq)
        skel.setAccelerations(ddq)

        # Clear external forces
        skel.clearExternalForces()

        # Estimate and apply GRF
        grf = estimate_grf(skel, foot_bodies, mass_kg, ground_y=0.0,
                           contact_threshold=args.contact_threshold)
        any_contact = False
        for ci, body in enumerate(foot_bodies):
            wrench = grf[body.getName()]
            if np.any(wrench != 0):
                any_contact = True
                contact_all[t, ci] = 1.0
                # Apply force at foot world position
                world_pos = body.getWorldTransform().translation()
                force = wrench[3:]   # [fx, fy, fz]
                torque = wrench[:3]  # [tx, ty, tz]
                body.addExtForce(force, world_pos, False, False)
                if np.any(torque != 0):
                    body.addExtTorque(torque, False)
            grf_all[t, ci*6:(ci+1)*6] = wrench

        if any_contact:
            n_contact_frames += 1

        # Compute ID with external forces applied
        # getInverseDynamics(accs) = M*ddq + C(q,dq) - G - F_ext
        torques_all[t] = skel.getInverseDynamics(ddq)

        # Compute root residual: M*ddq + C - tau_ext - tau_id should = 0
        # The root DOFs (0-5) carry the residual if GRF is imperfect
        residuals_all[t, :3] = torques_all[t, :3]   # root rotation residuals
        residuals_all[t, 3:] = torques_all[t, 3:6]  # root translation residuals (= forces)

    print(f'  Contact detected in {n_contact_frames}/{nframes} frames')

    # 8. Save outputs
    print('Saving results...')
    torque_cols = [f'{skel_dof_names[i]}_torque' for i in range(skel.getNumDofs())]
    write_sto(os.path.join(args.out, 'inverse_dynamics.sto'), times, torque_cols, torques_all)

    res_cols = ['res_rot_x', 'res_rot_y', 'res_rot_z', 'res_fx', 'res_fy', 'res_fz']
    write_sto(os.path.join(args.out, 'residuals.sto'), times, res_cols, residuals_all)

    grf_cols = []
    for b in foot_bodies:
        for comp in ['tx', 'ty', 'tz', 'fx', 'fy', 'fz']:
            grf_cols.append(f'{b.getName()}_{comp}')
    write_sto(os.path.join(args.out, 'estimated_grf.sto'), times, grf_cols, grf_all)

    # Summary statistics (exclude root DOFs 0-5 from torque summary)
    joint_torques = torques_all[:, 6:]
    body_weight = mass_kg * 9.81
    rms_res_force = np.sqrt(np.mean(residuals_all[:, 3:]**2))
    print(f'\nDone!')
    print(f'  Contact frames: {n_contact_frames}/{nframes}')
    print(f'  Mean |joint torque| (distal DOFs): {np.mean(np.abs(joint_torques)):.2f} N·m')
    print(f'  Root residual RMS force: {rms_res_force:.2f} N  '
          f'(body weight = {body_weight:.1f} N, ratio = {rms_res_force/body_weight:.2f})')
    print(f'  Outputs: {args.out}')


if __name__ == '__main__':
    main()
