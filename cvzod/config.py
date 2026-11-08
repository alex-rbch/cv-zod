from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Settings
from .io import sha256

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
METHODS = ('cv-zod', 'zod', 'ges', 'surgd')
EXPERIMENTS = ('fluid', 'molecules')
RATES = {'fluid': dict(zip(METHODS, (5., 5., 50., 50.))),
         'molecules': dict(zip(METHODS, (.001, .001, .005, .005)))}


def make_config(experiment, iterations=100, seeds=None, conformations=10, *,
                query_budget=16, norm_queries=2, discount=.7, radius=1e-4,
                rate_scale=1., hint_dims=(1,), seed_count=20, seed_start=1000):
    if experiment not in EXPERIMENTS:
        raise ValueError('Choose fluid or molecules.')
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError('The number of optimization steps must be a positive integer.')
    if seeds is None:
        if not isinstance(seed_count, int) or seed_count < 1 or not isinstance(seed_start, int):
            raise ValueError('Use a positive integer seed count and an integer seed start.')
        if seed_start < 0 or seed_start+seed_count > 2**32:
            raise ValueError('Seeds must be between zero and 2**32-1.')
        seeds = list(range(seed_start, seed_start+seed_count))
    else:
        seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Use distinct nonnegative integer seeds smaller than 2**32.')
    if not isinstance(conformations, int) or not 1 <= conformations <= 10:
        raise ValueError('Choose between one and ten conformations per molecule.')
    if not np.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError('The learning-rate scale must be finite and positive.')
    values = [hint_dims] if isinstance(hint_dims, int) else list(hint_dims)
    if not values or any(not isinstance(value, int) or value < 1 for value in values):
        raise ValueError('Hint dimensions must be positive integers.')
    values = list(dict.fromkeys(values))
    for value in values:
        Settings('cv-zod', RATES[experiment]['cv-zod']*rate_scale, history=value,
                 norm_queries=norm_queries, discount=discount, radius=radius,
                 query_budget=query_budget)
    files = sorted(path for path in PACKAGE.rglob('*') if path.is_file()
                   and not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(PACKAGE).parts))
    files += [ROOT/'run.py', ROOT/'requirements.txt']
    return dict(version=4, iterations=iterations, fluid_seeds=seeds,
                conformations=conformations, query_budget=query_budget,
                norm_queries=norm_queries, discount=discount, radius=radius,
                rate_scale=rate_scale, hint_dims={experiment: values},
                source_checksums={str(path.relative_to(ROOT)): sha256(path) for path in files})


def shared_config(config):
    return {key: value for key, value in config.items() if key != 'hint_dims'}


def config_hash(config):
    encoded = json.dumps(shared_config(config), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_key(experiment, method, seed, hint_dim=1, molecule=None, case_id=None):
    case = f'{molecule}_{case_id}' if experiment == 'molecules' else f's{seed}'
    suffix = f'_k{hint_dim}' if method == 'cv-zod' else ''
    return f'{experiment}_{case}_{method}{suffix}'


def jobs_for_experiment(experiment, config):
    if experiment == 'fluid':
        cases = [dict(seed=seed) for seed in config['fluid_seeds']]
    else:
        from .molecule_conformation.problem import preparation
        manifest, _ = preparation()
        cases = [{key: case[key] for key in ('seed', 'molecule', 'case_id')}
                 for case in manifest['cases'] if case['seed'] < config['conformations']]
    jobs = []
    for case in cases:
        for method, rate in RATES[experiment].items():
            for value in config['hint_dims'][experiment] if method == 'cv-zod' else (1,):
                settings = Settings(method, rate*config['rate_scale'], history=value,
                                    query_budget=config['query_budget'], norm_queries=config['norm_queries'],
                                    discount=config['discount'], radius=config['radius'])
                jobs.append(dict(key=run_key(experiment, method, hint_dim=value, **case),
                                 experiment=experiment, settings=asdict(settings), **case))
    return jobs


def figure_for(experiment, config):
    return 3 if experiment == 'molecules' else (4 if len(config['hint_dims'][experiment]) > 1 else 2)
