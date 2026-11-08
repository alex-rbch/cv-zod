import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import traceback

import numpy as np

from .config import EXPERIMENTS, ROOT, config_hash, figure_for, jobs_for_experiment, make_config, shared_config
from .core import Optimizer, Settings, alignment, digest_array
from .io import sha256, write_json


def load_completed(output, job, fingerprint):
    path = output/'data'/(job['key']+'.json')
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data['config_sha256'] != fingerprint or any(data[key] != value for key, value in job.items()):
        raise ValueError('Saved run settings differ: '+job['key'])
    if data['status'] != 'complete':
        return None
    if sha256(path.with_suffix('.npz')) != data['arrays_sha256']:
        raise ValueError('Saved run arrays were modified: '+job['key'])
    return data


def run_one(output, job):
    output = Path(output)
    config = json.loads((output/'config.json').read_text())
    fingerprint = config_hash(config)
    if job['experiment'] == 'fluid':
        from .fluid_dynamics.problem import FluidProblem
        problem = FluidProblem()
        surrogate = None if job['settings']['method'] == 'zod' else FluidProblem(16)
        initial = np.random.default_rng(job['seed']).normal(0, .005, 8192)
        provenance = dict(initialization='paired Gaussian velocity perturbation')
    else:
        from .molecule_conformation.problem import make_problems
        problem, surrogate, initial, provenance = make_problems(job['molecule'], job['case_id'])
    optimizer = Optimizer(problem, surrogate, initial, Settings(**job['settings']), job['seed'])
    path = output/'data'/(job['key']+'.json')
    data = dict(job, status='running', config_sha256=fingerprint, initial_sha256=digest_array(initial),
                provenance=provenance, iterations=[], diagnostics=[],
                started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for step in range(config['iterations']):
            record, before, basis, columns, fresh, project = optimizer.step()
            if data['iterations']:
                data['iterations'][-1]['objective_after'] = record['objective_before']
            data['iterations'].append(record)
            if job['settings']['method'] in ('cv-zod', 'ges'):
                length = job['settings']['history']
                nested = range(1, length+1) if job['experiment'] == 'fluid' and length > 1 else ()
                diagnostic = alignment(problem, before, basis, columns, fresh, project,
                                       nested=nested)
                diagnostic['iteration'] = step
                data['diagnostics'].append(diagnostic)
            if (step+1) % 10 == 0:
                write_json(path, data)
        final = optimizer.oracle.final(optimizer.x)
        data['iterations'][-1]['objective_after'] = final
        arrays = dict(initial=initial, final=optimizer.x)
        if job['experiment'] == 'fluid' and job['seed'] == config['fluid_seeds'][0]:
            arrays.update(terminal_density=problem.current_density(), initial_density=problem.initial_density,
                          target_density=problem.target_density)
        np.savez_compressed(path.with_suffix('.npz'), **arrays)
        data.update(status='complete', final_objective=final, optimization_queries=optimizer.oracle.optimization_total,
                    final_reporting_calls=optimizer.oracle.reporting_total, actual_objective_calls=optimizer.oracle.total,
                    arrays_sha256=sha256(path.with_suffix('.npz')))
        write_json(path, data)
    except Exception:
        data.update(status='failed', error=traceback.format_exc())
        write_json(path, data)
        raise
    finally:
        for item in (problem, surrogate):
            if item is not None and hasattr(item, 'close'):
                item.close()
    return job['key']


def execute(experiment, output, workers=None, iterations=100, seeds=None, conformations=10,
            plot_only=False, **options):
    if experiment not in EXPERIMENTS:
        raise ValueError('Choose fluid or molecules.')
    if workers is not None and workers < 1:
        raise ValueError('The worker count must be positive.')
    output = Path(output).resolve()
    path = output/'config.json'
    if plot_only:
        if not path.exists():
            raise ValueError('No saved results found. Run the experiment first.')
        config = json.loads(path.read_text())
    else:
        config = make_config(experiment, iterations, seeds, conformations, **options)
        if path.exists():
            saved = json.loads(path.read_text())
            if shared_config(saved) != shared_config(config):
                raise ValueError('Saved run settings differ. Choose a new --output directory.')
            config['hint_dims'] = {**saved['hint_dims'], **config['hint_dims']}
        (output/'data').mkdir(parents=True, exist_ok=True)
        write_json(path, config)
    if experiment not in config['hint_dims']:
        raise ValueError('No saved '+experiment+' experiment found. Run it first.')
    jobs = jobs_for_experiment(experiment, config)
    fingerprint = config_hash(config)
    runs = {job['key']: load_completed(output, job, fingerprint) for job in jobs}
    pending = [job for job in jobs if runs[job['key']] is None]
    if plot_only and pending:
        raise ValueError('Some runs are incomplete. Run the experiment before using --plot-only.')
    print(f'{experiment}: {len(jobs)} runs; {len(jobs)-len(pending)} already complete.', flush=True)
    if pending:
        if experiment == 'fluid':
            from .fluid_dynamics.problem import native_library
            native_library()
        else:
            from .molecule_conformation.problem import model_path
            model_path()
        workers = min(4 if experiment == 'molecules' else 8, os.cpu_count() or 1) if workers is None else workers
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_one, output, job): job for job in pending}
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    key = future.result()
                    runs[key] = load_completed(output, futures[future], fingerprint)
                    print(f'Completed {count}/{len(pending)} runs.', flush=True)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    from .plots import render
    render(experiment, config, runs, output)
    print(f'Saved {output / f"figure{figure_for(experiment, config)}.pdf"}', flush=True)
    return config, runs


def main():
    parser = argparse.ArgumentParser(description='Run fluid dynamics or molecular conformations experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment', choices=tuple(EXPERIMENTS))
    parser.add_argument('--workers', type=int, help='Number of parallel runs; chosen automatically if omitted.')
    parser.add_argument('--output', type=Path, default=ROOT/'results', help='Directory for saved runs and graphs.')
    parser.add_argument('--steps', type=int, default=100, help='Optimization steps per run.')
    parser.add_argument('--seeds', type=int, default=20, help='Paired runs for fluid experiments.')
    parser.add_argument('--seed-start', type=int, default=1000, help='First fluid random seed.')
    parser.add_argument('--conformations', type=int, default=10, help='Starting conformations per molecule, from 1 to 10.')
    parser.add_argument('--query-budget', type=int, default=16, help='Objective queries per zeroth-order update, including the current value.')
    parser.add_argument('--norm-queries', type=int, default=2, help='Queries allocated to the two CV-ZOD norm estimates.')
    parser.add_argument('--discount', type=float, default=.7, help='Discount factor for norm estimates, in [0, 1).')
    parser.add_argument('--radius', type=float, default=1e-4, help='Finite-difference radius.')
    parser.add_argument('--rate-scale', type=float, default=1., help='Multiplier for all default learning rates.')
    parser.add_argument('--hint-dim', dest='hint_dims', type=int, nargs='+', default=[1], metavar='K',
                        help='CV-ZOD hint space dimensions k to compare: the current gradient plus up to k-1 stored earlier gradients. Only the supplied values are used.')
    parser.add_argument('--plot-only', action='store_true', help='Plot completed runs using their saved settings.')
    args = parser.parse_args()
    try:
        execute(args.experiment, args.output, args.workers, iterations=args.steps,
                conformations=args.conformations, plot_only=args.plot_only,
                seed_count=args.seeds, seed_start=args.seed_start,
                query_budget=args.query_budget, norm_queries=args.norm_queries,
                discount=args.discount, radius=args.radius, rate_scale=args.rate_scale,
                hint_dims=args.hint_dims)
    except ValueError as error:
        parser.error(str(error))
