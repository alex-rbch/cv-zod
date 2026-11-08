from functools import lru_cache
import json
from pathlib import Path
import urllib.request

import numpy as np

from ..core import digest_array
from ..io import sha256

HERE = Path(__file__).resolve().parent
MODEL_SHA256 = '165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f'


@lru_cache(maxsize=1)
def model_path():
    path = HERE.parent / '.cache/mace/MACE-OFF23_small.model'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        url = 'https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true'
        print('Downloading the Figure 3 model...', flush=True)
        try:
            urllib.request.urlretrieve(url, temporary)
            if sha256(temporary) != MODEL_SHA256:
                raise RuntimeError('Downloaded model checksum does not match MACE-OFF23-small.')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) != MODEL_SHA256:
        raise RuntimeError('Cached model checksum does not match MACE-OFF23-small.')
    return path


@lru_cache(maxsize=1)
def calculator():
    import torch
    from mace.calculators import MACECalculator
    torch.set_num_threads(1)
    model = model_path()
    return MACECalculator(model_paths=str(model), device='cpu', default_dtype='float64')


class MolecularProblem:
    def __init__(self, symbols, initial):
        from ase import Atoms
        self.shape = initial.shape
        self.atoms = Atoms(symbols=symbols, positions=initial)
        self.calculator = calculator()

    def positions(self, x):
        positions = np.asarray(x, dtype=np.float64).reshape(self.shape)
        if not np.isfinite(positions).all():
            raise FloatingPointError('Invalid molecular coordinates')
        return positions

    def evaluate(self, x, force=False):
        import torch
        self.atoms.positions[:] = self.positions(x)
        batch = self.calculator._atoms_to_batch(self.atoms)
        with torch.set_grad_enabled(force):
            result = self.calculator.models[0](batch.to_dict(), training=False, compute_force=force,
                                               compute_virials=False, compute_stress=False)
        return result

    def objective(self, x):
        value = float(self.evaluate(x)['energy'].detach().cpu().item())
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite MACE energy')
        return value

    def gradient(self, x):
        return -self.evaluate(x, force=True)['forces'].detach().cpu().numpy().reshape(-1)

    def projector(self, x):
        positions = self.positions(x)
        centered = positions - positions.mean(axis=0, keepdims=True)
        modes = np.stack([np.cross(axis, centered).reshape(-1) for axis in np.eye(3)], axis=1)
        left, values, _ = np.linalg.svd(modes, full_matrices=False)
        rank = int(np.count_nonzero(values > np.finfo(float).eps * max(modes.shape) * values[0])) if values.size else 0
        rotations = left[:, :rank]

        def project(directions):
            values = np.asarray(directions, dtype=np.float64).reshape(self.shape[0], 3, -1).copy()
            values -= values.mean(axis=0, keepdims=True)
            flat = values.reshape(positions.size, -1)
            flat -= np.einsum('ij,jk->ik', rotations, np.einsum('ij,ik->jk', rotations, flat))
            return flat
        return project


class MMFF:
    def __init__(self, topology, initial):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from ase.units import kcal, mol
        self.conversion = float(kcal / mol)
        self.topology = Chem.Mol(topology)
        conformer = self.topology.GetConformer()
        for i, row in enumerate(initial):
            conformer.SetAtomPosition(i, tuple(float(v) for v in row))
        self.properties = AllChem.MMFFGetMoleculeProperties(self.topology, mmffVariant='MMFF94')
        self.force_field = AllChem.MMFFGetMoleculeForceField(self.topology, self.properties)
        if self.force_field is None:
            raise ValueError('Missing MMFF94 force field')

    def gradient(self, x):
        positions = tuple(np.asarray(x, dtype=float))
        self.force_field.CalcEnergy(positions)
        result = np.asarray(self.force_field.CalcGrad(positions), dtype=np.float64) * self.conversion
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite MMFF94 gradient')
        return result


@lru_cache(maxsize=1)
def preparation():
    manifest_path = HERE / 'data/preparation.json'
    coordinates_path = HERE / 'data/coordinates.npz'
    manifest = json.loads(manifest_path.read_text())
    if sha256(coordinates_path) != manifest['coordinates_sha256']:
        raise ValueError('Frozen coordinates were modified')
    with np.load(coordinates_path, allow_pickle=False) as arrays:
        coordinates = {key: arrays[key] for key in arrays.files}
    for case in manifest['cases']:
        initial = coordinates[case['initial_key']]
        if digest_array(initial) != case['initial_coordinates_sha256']:
            raise ValueError('A frozen initial conformation changed')
        initial.setflags(write=False)
    return manifest, coordinates


def make_problems(molecule, case_id):
    from rdkit import Chem
    manifest, coordinates = preparation()
    case, = [case for case in manifest['cases'] if case['molecule'] == molecule and case['case_id'] == case_id]
    topology_record = manifest['topologies'][molecule]
    topology = Chem.MolFromMolBlock(topology_record['molblock'], sanitize=True, removeHs=False)
    initial = coordinates[case['initial_key']].copy()
    provenance = dict(case, coordinate_archive_sha256=manifest['coordinates_sha256'],
                      symbols=topology_record['symbols'],
                      objective='MACE-OFF23-small', units='eV', surrogate='MMFF94',
                      surrogate_conversion_factor=.04336410390059322,
                      model_sha256=MODEL_SHA256)
    return MolecularProblem(topology_record['symbols'], initial), MMFF(topology, initial), initial.ravel(), provenance
