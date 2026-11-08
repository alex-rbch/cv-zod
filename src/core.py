from collections import Counter, deque
from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Settings:
    method: str
    rate: float
    history: int = 1
    norm_queries: int = 2
    discount: float = .7
    radius: float = 1e-4
    multiplier_cap: float = 10.
    query_budget: int = 16

    def __post_init__(self):
        if self.method not in ('cv-zod', 'zod', 'ges', 'surgd'):
            raise ValueError('Unknown method')
        if not np.isfinite([self.rate, self.radius, self.discount, self.multiplier_cap]).all():
            raise ValueError('Optimization settings must be finite')
        if self.rate <= 0 or self.radius <= 0 or not 0 <= self.discount < 1 or self.multiplier_cap < 1:
            raise ValueError('Invalid optimization settings')
        if not isinstance(self.query_budget, int) or self.query_budget < 2:
            raise ValueError('The query budget must be an integer of at least two')
        if self.method == 'cv-zod':
            if not isinstance(self.history, int) or not isinstance(self.norm_queries, int):
                raise ValueError('History length and norm queries must be integers')
            if self.history < 1 or self.norm_queries < 2:
                raise ValueError('Use a positive history length and at least two norm queries')
            if self.history+self.norm_queries+2 > self.query_budget:
                raise ValueError('The query budget must cover the current value, history projection, norm queries, and at least one update query')


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def basis_for(columns):
    if not columns.shape[1]:
        return columns.copy(), dict(rank=0, singular_values=[], orthogonality_error=0.)
    norms = np.linalg.norm(columns, axis=0)
    normalized = np.divide(columns, norms, out=np.zeros_like(columns), where=norms > 0)
    left, values, _ = np.linalg.svd(normalized, full_matrices=False)
    rank = int(np.count_nonzero(values > 1e-6 * values[0]))
    basis = left[:, :rank].copy()
    if rank:
        pivots = np.argmax(np.abs(basis), axis=0)
        basis *= np.sign(basis[pivots, np.arange(rank)])
    error = float(np.linalg.norm(np.einsum('ij,ik->jk', basis, basis) - np.eye(rank)))
    if error > 1e-10:
        raise ArithmeticError('Nonorthogonal hint basis')
    return basis, dict(rank=rank, singular_values=values.tolist(), orthogonality_error=error)


class Oracle:
    def __init__(self, problem, budget=16):
        self.problem, self.budget = problem, budget
        self.total = 0
        self.optimization_total = 0
        self.reporting_total = 0
        self.counts = Counter()
        self.finished = False

    def begin(self):
        if self.finished:
            raise RuntimeError('Optimization after final report')
        self.counts.clear()

    def query(self, x, phase):
        if phase not in ('center', 'projection', 'norm_full', 'norm_residual', 'update'):
            raise ValueError('Unclassified query')
        if self.finished or sum(self.counts.values()) >= self.budget:
            raise RuntimeError('Query budget exceeded')
        self.counts[phase] += 1
        self.total += 1
        self.optimization_total += 1
        value = float(self.problem.objective(x))
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite objective')
        return value

    def final(self, x):
        if self.finished:
            raise RuntimeError('Duplicate final report')
        self.finished = True
        self.total += 1
        self.reporting_total += 1
        value = float(self.problem.objective(x))
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite final objective')
        return value


class Optimizer:
    def __init__(self, problem, surrogate, initial, settings, seed):
        self.problem, self.surrogate, self.settings = problem, surrogate, settings
        self.x = np.asarray(initial, dtype=np.float64).ravel().copy()
        self.parameters32 = self.x.astype(np.float32)
        self.rng = np.random.RandomState(seed)
        self.oracle = Oracle(problem, settings.query_budget)
        self.history = deque(maxlen=settings.history)
        self.iteration = 0
        self.full_sum = self.residual_sum = 0.
        if settings.method != 'zod' and surrogate is None:
            raise ValueError('A surrogate is required')

    def derivative(self, directions, center, phase):
        return np.array([(self.oracle.query(self.x + self.settings.radius * direction, phase) - center)
                         / self.settings.radius for direction in directions.T])

    def gaussian(self, number, project):
        return project(self.rng.standard_normal((self.x.size, number)))

    def step(self):
        settings = self.settings
        self.oracle.begin()
        center = self.oracle.query(self.x, 'center')
        before = self.x.copy()
        project = self.problem.projector(self.x)
        fresh = None
        if settings.method != 'zod':
            fresh = np.asarray(project(self.surrogate.gradient(self.x)[:, None])[:, 0], dtype=np.float64)
            if not np.isfinite(fresh).all():
                raise FloatingPointError('Nonfinite surrogate gradient')
        basis = np.empty((self.x.size, 0))
        columns = basis.copy()
        rank_info = dict(rank=0, singular_values=[], orthogonality_error=0.)
        coefficients = np.empty(0)
        norm_full = norm_residual = None
        factor = 1.
        if settings.method == 'cv-zod':
            immutable = fresh.copy()
            immutable.setflags(write=False)
            self.history.append((self.iteration, immutable, digest_array(immutable)))
            columns = np.column_stack([entry[1] for entry in reversed(self.history)])
            basis, rank_info = basis_for(columns)
            coefficients = self.derivative(basis, center, 'projection')
            reference = np.einsum('ij,j->i', basis, coefficients)
            full = self.gaussian((settings.norm_queries + 1) // 2, project)
            residual = self.gaussian(settings.norm_queries // 2, project)
            residual -= np.einsum('ij,jk->ik', basis, np.einsum('ij,ik->jk', basis, residual))
            norm_full = float(np.mean(self.derivative(full, center, 'norm_full') ** 2))
            norm_residual = float(np.mean(self.derivative(residual, center, 'norm_residual') ** 2))
            self.full_sum = settings.discount * self.full_sum + norm_full
            self.residual_sum = settings.discount * self.residual_sum + norm_residual
            ratio = self.full_sum / self.residual_sum if self.residual_sum > 0 else (np.inf if self.full_sum > 0 else 0.)
            factor = float(np.clip(ratio, 1., settings.multiplier_cap))
        else:
            reference = fresh.copy() if settings.method == 'surgd' else np.zeros_like(self.x)
        batch = 0
        if settings.method != 'surgd':
            batch = self.oracle.budget - sum(self.oracle.counts.values())
            if batch < 1:
                raise ValueError('No update queries remain')
            directions = self.rng.standard_normal((self.x.size, batch))
            if settings.method == 'ges':
                norm = np.linalg.norm(fresh)
                if norm:
                    weights = self.rng.standard_normal(batch)
                    directions = (directions / np.sqrt(self.x.size) + np.outer(fresh / norm, weights)) / np.sqrt(2.)
            directions = project(directions)
            samples = self.derivative(directions, center, 'update') - np.einsum('ij,i->j', directions, reference)
            estimate = reference + np.einsum('ij,j->i', directions, samples / batch)
            if sum(self.oracle.counts.values()) != self.oracle.budget:
                raise AssertionError('A zeroth-order update must use its full query budget')
        else:
            estimate = reference
        if not np.isfinite(estimate).all():
            raise FloatingPointError('Nonfinite update')
        rate = settings.rate * factor
        self.parameters32 -= np.float32(rate) * estimate.astype(np.float32)
        self.x = self.parameters32.astype(np.float64)
        record = dict(iteration=self.iteration, objective_before=center, objective_after=None,
                      learning_rate=rate, adaptive_factor=factor, query_counts=dict(self.oracle.counts),
                      optimization_queries=sum(self.oracle.counts.values()), update_directions=batch,
                      norm_full_estimate=norm_full, norm_residual_estimate=norm_residual,
                      full_running_sum=self.full_sum, residual_running_sum=self.residual_sum,
                      projection_coefficients=coefficients.tolist(),
                      history_iterations=[i for i, _, _ in reversed(self.history)],
                      history_gradient_sha256=[digest for _, _, digest in reversed(self.history)],
                      fresh_surrogate_gradients=int(fresh is not None), **rank_info)
        self.iteration += 1
        return record, before, basis, columns, fresh, project


def alignment(problem, before, basis, columns, fresh, project, nested=()):
    true = project(problem.gradient(before)[:, None])[:, 0]
    true_norm = float(np.linalg.norm(true))
    surrogate_norm = float(np.linalg.norm(fresh))
    valid = true_norm > 0 and surrogate_norm > 0
    signed = float(np.clip(np.dot(fresh, true) / (true_norm * surrogate_norm), -1., 1.)) if valid else 0.
    cosine = float(np.clip(np.linalg.norm(np.einsum('ij,i->j', basis, true)) / true_norm, 0., 1.)) if basis.shape[1] and true_norm else abs(signed)
    result = dict(cosine=cosine, signed_surrogate_cosine=signed, angle_degrees=float(np.degrees(np.arccos(cosine))) if valid else None,
                  valid=valid, true_gradient_norm=true_norm, surrogate_gradient_norm=surrogate_norm)
    if nested:
        result['nested_cosines'] = {}
        values = {columns.shape[1]: cosine}
        for k in nested:
            width = min(k, columns.shape[1])
            if width not in values:
                q, _ = basis_for(columns[:, :width])
                values[width] = float(np.clip(np.linalg.norm(np.einsum('ij,i->j', q, true)) / true_norm, 0., 1.)) if true_norm else 0.
            result['nested_cosines'][str(k)] = values[width]
    return result
