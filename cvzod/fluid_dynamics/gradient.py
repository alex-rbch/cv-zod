import math
from dataclasses import dataclass
import numpy as np
from PIL import Image
import warp as wp

@dataclass(frozen=True)
class FluidSettings:
    n_grid: int = 32
    pressure_iterations: int = 25

    def __post_init__(self) -> None:
        if self.n_grid < 1:
            raise ValueError('n_grid must be at least 1')
        if self.pressure_iterations < 1:
            raise ValueError('pressure_iterations must be at least 1')

    @property
    def dh(self) -> float:
        return 1.0 / float(self.n_grid)

@wp.func
def cyclic_index(idx: int, n_grid: int):
    ret_idx = idx % n_grid
    if ret_idx < 0:
        ret_idx += n_grid
    return ret_idx

@wp.kernel
def advect(dt: float, vx: wp.array2d[float], vy: wp.array2d[float], f0: wp.array2d[float], n_grid: int, f1: wp.array2d[float]):
    (i, j) = wp.tid()
    center_xs = float(i) - vx[i, j] * dt
    center_ys = float(j) - vy[i, j] * dt
    left_idx = int(wp.floor(center_xs))
    bot_idx = int(wp.floor(center_ys))
    s1 = center_xs - float(left_idx)
    s0 = 1.0 - s1
    t1 = center_ys - float(bot_idx)
    t0 = 1.0 - t1
    i0 = cyclic_index(left_idx, n_grid)
    i1 = cyclic_index(left_idx + 1, n_grid)
    j0 = cyclic_index(bot_idx, n_grid)
    j1 = cyclic_index(bot_idx + 1, n_grid)
    f1[i, j] = s0 * (t0 * f0[i0, j0] + t1 * f0[i0, j1]) + s1 * (t0 * f0[i1, j0] + t1 * f0[i1, j1])

@wp.kernel
def divergence(wx: wp.array2d[float], wy: wp.array2d[float], n_grid: int, dh: float, div: wp.array2d[float]):
    (i, j) = wp.tid()
    div[i, j] = 0.5 * (wx[cyclic_index(i + 1, n_grid), j] - wx[cyclic_index(i - 1, n_grid), j] + wy[i, cyclic_index(j + 1, n_grid)] - wy[i, cyclic_index(j - 1, n_grid)]) / dh

@wp.kernel
def jacobi_iter(div: wp.array2d[float], p0: wp.array2d[float], n_grid: int, dh: float, p1: wp.array2d[float]):
    (i, j) = wp.tid()
    p1[i, j] = 0.25 * (-dh * dh * div[i, j] + p0[cyclic_index(i - 1, n_grid), j] + p0[cyclic_index(i + 1, n_grid), j] + p0[i, cyclic_index(j - 1, n_grid)] + p0[i, cyclic_index(j + 1, n_grid)])

@wp.kernel
def update_velocities(p: wp.array2d[float], wx: wp.array2d[float], wy: wp.array2d[float], n_grid: int, dh: float, vx: wp.array2d[float], vy: wp.array2d[float]):
    (i, j) = wp.tid()
    vx[i, j] = wx[i, j] - 0.5 * (p[cyclic_index(i + 1, n_grid), j] - p[cyclic_index(i - 1, n_grid), j]) / dh
    vy[i, j] = wy[i, j] - 0.5 * (p[i, cyclic_index(j + 1, n_grid)] - p[i, cyclic_index(j - 1, n_grid)]) / dh

@wp.kernel
def compute_loss(actual_state: wp.array2d[float], target_state: wp.array2d[float], n_grid: int, loss: wp.array[float]):
    (i, j) = wp.tid()
    loss_value = (actual_state[i, j] - target_state[i, j]) * (actual_state[i, j] - target_state[i, j]) / float(n_grid * n_grid)
    wp.atomic_add(loss, 0, loss_value)

class WarpGradient:

    def __init__(self, sim_steps=1000, initial_image_path=None, target_image_path=None, settings=None):
        self._settings = settings if settings is not None else FluidSettings()
        self._grid_shape = (self._settings.n_grid, self._settings.n_grid)
        self.pressure_arrays = []
        self.wx_arrays = []
        self.wy_arrays = []
        self.vx_arrays = []
        self.vy_arrays = []
        self.density_arrays = []
        self.div_arrays = []
        self.segment_size = math.ceil(math.sqrt(sim_steps))
        self.num_segments = math.ceil(sim_steps / self.segment_size)
        self.sim_steps = self.segment_size * self.num_segments
        self.pressure_iterations = self._settings.pressure_iterations
        self.dt = 1.0
        for _step in range(self.segment_size + 1):
            self.vx_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
            self.vy_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
            self.density_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
        for _step in range(self.segment_size):
            self.wx_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
            self.wy_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
            self.div_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
            for _iter in range(self.pressure_iterations):
                self.pressure_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
        self.pressure_arrays.append(wp.zeros(self.grid_shape, dtype=float, requires_grad=True))
        self.segment_start_vx_arrays = []
        self.segment_start_vy_arrays = []
        self.segment_start_density_arrays = []
        self.segment_start_pressure_arrays = []
        for _segment_index in range(self.num_segments):
            self.segment_start_vx_arrays.append(wp.zeros(self.grid_shape, dtype=float))
            self.segment_start_vy_arrays.append(wp.zeros(self.grid_shape, dtype=float))
            self.segment_start_density_arrays.append(wp.zeros(self.grid_shape, dtype=float))
            self.segment_start_pressure_arrays.append(wp.zeros(self.grid_shape, dtype=float))
        self.vx_array_grad_saved = wp.zeros(self.grid_shape, dtype=float)
        self.vy_array_grad_saved = wp.zeros(self.grid_shape, dtype=float)
        self.density_array_grad_saved = wp.zeros(self.grid_shape, dtype=float)
        self.pressure_array_grad_saved = wp.zeros(self.grid_shape, dtype=float)
        self._target_density = None
        self._initial_density = None
        self.target_wp = wp.zeros(self.grid_shape, dtype=float)
        self.set_target(self._load_density_image(target_image_path, 'target'))
        self.set_density(self._load_density_image(initial_image_path, 'initial'))
        self.loss = wp.zeros((1,), dtype=float, requires_grad=True)
        self.tape = None

    def clear_tape(self) -> None:
        if self.tape is not None:
            self.tape.zero()

    @property
    def settings(self) -> FluidSettings:
        return self._settings

    @property
    def grid_shape(self):
        return self._grid_shape

    def _load_density_image(self, image_path, image_label):
        if image_path is None:
            raise ValueError(f'{image_label}_image_path must be provided')
        image_base = Image.open(image_path).convert('L')
        image_resized = image_base.resize(self.grid_shape)
        return np.asarray(image_resized, dtype=np.float32) / 255.0

    def set_target(self, target_density) -> None:
        target_array = np.asarray(target_density, dtype=np.float32)
        if target_array.shape != self.grid_shape:
            raise ValueError(f'target shape {target_array.shape} does not match grid shape {self.grid_shape}')
        self._target_density = np.ascontiguousarray(target_array)
        wp.copy(self.target_wp, wp.array(self._target_density, dtype=float, device=wp.get_device()))

    def set_density(self, density) -> None:
        density_array = np.asarray(density, dtype=np.float32)
        if density_array.shape != self.grid_shape:
            raise ValueError(f'density shape {density_array.shape} does not match grid shape {self.grid_shape}')
        self._initial_density = np.ascontiguousarray(density_array)
        wp.copy(self.density_arrays[0], wp.array(self._initial_density, dtype=float, device=wp.get_device()))

    def reset_initial_state(self) -> None:
        wp.copy(self.density_arrays[0], wp.array(self._initial_density, dtype=float, device=wp.get_device()))
        self.pressure_arrays[0].zero_()

    def step(self, step_index) -> None:
        wp.launch(advect, self.grid_shape, inputs=[self.dt, self.vx_arrays[step_index - 1], self.vy_arrays[step_index - 1], self.vx_arrays[step_index - 1], self.settings.n_grid], outputs=[self.wx_arrays[step_index - 1]])
        wp.launch(advect, self.grid_shape, inputs=[self.dt, self.vx_arrays[step_index - 1], self.vy_arrays[step_index - 1], self.vy_arrays[step_index - 1], self.settings.n_grid], outputs=[self.wy_arrays[step_index - 1]])
        wp.launch(divergence, self.grid_shape, inputs=[self.wx_arrays[step_index - 1], self.wy_arrays[step_index - 1], self.settings.n_grid, self.settings.dh], outputs=[self.div_arrays[step_index - 1]])
        for k in range(self.pressure_iterations):
            input_index = self.pressure_iterations * (step_index - 1) + k
            output_index = input_index + 1
            wp.launch(jacobi_iter, self.grid_shape, inputs=[self.div_arrays[step_index - 1], self.pressure_arrays[input_index], self.settings.n_grid, self.settings.dh], outputs=[self.pressure_arrays[output_index]])
        wp.launch(update_velocities, self.grid_shape, inputs=[self.pressure_arrays[output_index], self.wx_arrays[step_index - 1], self.wy_arrays[step_index - 1], self.settings.n_grid, self.settings.dh], outputs=[self.vx_arrays[step_index], self.vy_arrays[step_index]])
        wp.launch(advect, self.grid_shape, inputs=[self.dt, self.vx_arrays[step_index], self.vy_arrays[step_index], self.density_arrays[step_index - 1], self.settings.n_grid], outputs=[self.density_arrays[step_index]])

    def forward(self) -> None:
        self.reset_initial_state()
        self.loss.zero_()
        for segment_index in range(self.num_segments):
            wp.copy(self.segment_start_vx_arrays[segment_index], self.vx_arrays[0])
            wp.copy(self.segment_start_vy_arrays[segment_index], self.vy_arrays[0])
            wp.copy(self.segment_start_density_arrays[segment_index], self.density_arrays[0])
            wp.copy(self.segment_start_pressure_arrays[segment_index], self.pressure_arrays[0])
            for t in range(1, self.segment_size + 1):
                self.step(t)
            if segment_index < self.num_segments - 1:
                wp.copy(self.vx_arrays[0], self.vx_arrays[-1])
                wp.copy(self.vy_arrays[0], self.vy_arrays[-1])
                wp.copy(self.density_arrays[0], self.density_arrays[-1])
                wp.copy(self.pressure_arrays[0], self.pressure_arrays[-1])
        wp.launch(compute_loss, self.grid_shape, inputs=[self.density_arrays[self.segment_size], self.target_wp, self.settings.n_grid], outputs=[self.loss])

    def backward(self) -> None:
        for segment_index in range(self.num_segments - 1, -1, -1):
            wp.copy(self.vx_arrays[0], self.segment_start_vx_arrays[segment_index])
            wp.copy(self.vy_arrays[0], self.segment_start_vy_arrays[segment_index])
            wp.copy(self.density_arrays[0], self.segment_start_density_arrays[segment_index])
            wp.copy(self.pressure_arrays[0], self.segment_start_pressure_arrays[segment_index])
            with wp.Tape() as self.tape:
                for t in range(1, self.segment_size + 1):
                    self.step(t)
            if segment_index == self.num_segments - 1:
                self.loss.grad.fill_(1.0)
                wp.launch(compute_loss, self.grid_shape, inputs=[self.density_arrays[self.segment_size], self.target_wp, self.settings.n_grid], outputs=[self.loss], adj_inputs=[self.density_arrays[self.segment_size].grad, None, None], adj_outputs=[self.loss.grad], adjoint=True)
            else:
                wp.copy(self.vx_arrays[-1].grad, self.vx_array_grad_saved)
                wp.copy(self.vy_arrays[-1].grad, self.vy_array_grad_saved)
                wp.copy(self.density_arrays[-1].grad, self.density_array_grad_saved)
                wp.copy(self.pressure_arrays[-1].grad, self.pressure_array_grad_saved)
            self.tape.backward()
            if segment_index > 0:
                wp.copy(self.vx_array_grad_saved, self.vx_arrays[0].grad)
                wp.copy(self.vy_array_grad_saved, self.vy_arrays[0].grad)
                wp.copy(self.density_array_grad_saved, self.density_arrays[0].grad)
                wp.copy(self.pressure_array_grad_saved, self.pressure_arrays[0].grad)
                self.tape.zero()

    def _set_initial_velocity_parameters(self, parameters) -> None:
        wp.copy(self.vx_arrays[0], wp.array(parameters[0], dtype=float, device=wp.get_device()))
        wp.copy(self.vy_arrays[0], wp.array(parameters[1], dtype=float, device=wp.get_device()))

    def gradient_at(self, parameters):
        self.clear_tape()
        self._set_initial_velocity_parameters(parameters)
        self.forward()
        self.backward()
        gradients = [self.vx_arrays[0].grad.numpy().copy(), self.vy_arrays[0].grad.numpy().copy()]
        self.clear_tape()
        self._set_initial_velocity_parameters(parameters)
        return gradients
