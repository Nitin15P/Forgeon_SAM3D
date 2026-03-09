import nimblephysics as nimble
from typing import Dict, List, Tuple, Optional
import os
import tempfile
import shutil
import numpy as np
import subprocess

try:
    from nimblephysics.loader import absPath
except ImportError:
    absPath = None


def _find_geometry_folder(geometry_folder_path: Optional[str]) -> Optional[str]:
    """Resolve the Geometry folder path for OpenSim model meshes."""
    if geometry_folder_path and os.path.isdir(geometry_folder_path):
        return os.path.realpath(geometry_folder_path)
    if absPath:
        for rel in ['Geometry', '../../data/Geometry', '../../../data/Geometry']:
            cand = absPath(rel)
            if cand and os.path.isdir(cand):
                return os.path.realpath(cand)
    return None


def scale_opensim_model(unscaled_generic_osim_text: str,
                        skel: nimble.dynamics.Skeleton,
                        mass_kg: float,
                        height_m: float,
                        markers: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]],
                        overwrite_inertia: bool = False,
                        geometry_folder_path: Optional[str] = None) -> str:
    marker_names: List[str] = []
    if skel is not None:
        print('Adjusting marker locations on scaled OpenSim file', flush=True)
        body_scales_map: Dict[str, np.ndarray] = {}
        for i in range(skel.getNumBodyNodes()):
            body_node: nimble.dynamics.BodyNode = skel.getBodyNode(i)
            # Now that we adjust the markers BEFORE we rescale the body, we don't want to rescale the marker locations
            # at all.
            body_scales_map[body_node.getName()] = np.ones(3)
        marker_offsets_map: Dict[str, Tuple[str, np.ndarray]] = {}
        for k in markers:
            v = markers[k]
            marker_offsets_map[k] = (v[0].getName(), v[1])
            marker_names.append(k)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        if not tmpdirname.endswith('/'):
            tmpdirname += '/'

        # Ensure Geometry folder is available so OpenSim ScaleTool can load mesh files
        geom_src = _find_geometry_folder(geometry_folder_path)
        if geom_src:
            geom_dst = tmpdirname + 'Geometry'
            try:
                os.symlink(geom_src, geom_dst)
            except OSError:
                shutil.copytree(geom_src, geom_dst)

        # 9.1. Write the unscaled OpenSim file to disk
        unscaled_generic_osim_path = tmpdirname + 'unscaled_generic.osim'
        with open(unscaled_generic_osim_path, 'w') as f:
            f.write(unscaled_generic_osim_text)

        nimble.biomechanics.OpenSimParser.moveOsimMarkers(
            unscaled_generic_osim_path,
            body_scales_map,
            marker_offsets_map,
            tmpdirname + 'unscaled_but_with_optimized_markers.osim')

        # Optional: skip the OpenSim ScaleTool and just use the marker-adjusted OSIM.
        # This is useful when meshes or opensim-cmd are problematic but IK results are still desired.
        if os.environ.get('AB_SKIP_SCALETOOL', '0') == '1':
            # If we're skipping scaling, just return the marker-adjusted file as our "scaled" model.
            with open(tmpdirname + 'unscaled_but_with_optimized_markers.osim') as f:
                return '\n'.join(f.readlines())

        # 9.3. Write the XML instructions for the OpenSim scaling tool
        nimble.biomechanics.OpenSimParser.saveOsimScalingXMLFile(
            'optimized_scale_and_markers',
            skel,
            mass_kg,
            height_m,
            'unscaled_but_with_optimized_markers.osim',
            'Unassigned',
            'optimized_scale_and_markers.osim',
            tmpdirname + 'rescaling_setup.xml')

        # 9.4. Call the OpenSim scaling tool (requires opensim-cmd from conda)
        import shutil as _shutil
        opensim_cmd = _shutil.which('opensim-cmd')
        opensim_conda_env = os.environ.get('OPENSIM_CONDA_ENV', 'opensim_cmd')
        if not opensim_cmd:
            # Try OPENSIM_BIN_PATH (for venv users who add conda bin to PATH)
            opensim_bin = os.environ.get('OPENSIM_BIN_PATH')
            if opensim_bin:
                cand = os.path.join(opensim_bin, 'opensim-cmd') if os.path.isdir(opensim_bin) else opensim_bin
                if os.path.isfile(cand):
                    opensim_cmd = cand
            # Try conda env's bin
            if not opensim_cmd:
                for prefix in [os.environ.get('CONDA_PREFIX'), os.path.expanduser('~/miniconda3/envs/' + opensim_conda_env), os.path.expanduser('~/anaconda3/envs/' + opensim_conda_env)]:
                    if prefix and os.path.isdir(prefix):
                        cand = os.path.join(prefix, 'bin', 'opensim-cmd')
                        if os.path.isfile(cand):
                            opensim_cmd = cand
                            break
            # Try conda env directly (common locations)
            if not opensim_cmd and _shutil.which('conda'):
                conda_run = subprocess.run(
                    ['conda', 'run', '-n', opensim_conda_env, 'which', 'opensim-cmd'],
                    capture_output=True, text=True, timeout=10
                )
                if conda_run.returncode == 0 and conda_run.stdout.strip():
                    opensim_cmd = conda_run.stdout.strip()
        if not opensim_cmd:
            raise FileNotFoundError(
                'opensim-cmd not found. Install OpenSim via conda:\n'
                '  conda create -n opensim_cmd -c opensim-org opensim\n'
                'Then either: set OPENSIM_BIN_PATH to that env\'s bin/, or ensure conda is on PATH.'
            )
        command = f'cd {tmpdirname} && {opensim_cmd} run-tool {tmpdirname}rescaling_setup.xml'
        print('Scaling OpenSim files: ' + command, flush=True)
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) as p:
            for line in iter(p.stdout.readline, b''):
                print(line.decode(), end='', flush=True)
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f'opensim-cmd exited with code {p.returncode}')

        # 9.5. Overwrite the inertia properties of the resulting OpenSim skeleton file
        if overwrite_inertia:
            nimble.biomechanics.OpenSimParser.replaceOsimInertia(
                tmpdirname + 'optimized_scale_and_markers.osim',
                skel,
                tmpdirname + 'output_scaled.osim')
        else:
            shutil.copyfile(tmpdirname + 'optimized_scale_and_markers.osim',
                            tmpdirname + 'output_scaled.osim')

        with open(tmpdirname + 'output_scaled.osim') as f:
            output_file_raw_text = '\n'.join(f.readlines())
    return output_file_raw_text
