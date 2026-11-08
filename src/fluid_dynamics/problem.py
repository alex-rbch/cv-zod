import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
LIBRARY = None


def native_library():
    global LIBRARY
    if LIBRARY is not None:
        return LIBRARY
    source = HERE / 'forward.cpp'
    cache = HERE.parent / '.cache' / 'native'
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    destination = cache / ('fluid_' + digest + ('.dylib' if platform.system() == 'Darwin' else '.so'))
    if not destination.exists():
        temporary = destination.with_name(destination.name + '.' + str(os.getpid()))
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if compiler is None:
            raise RuntimeError('Install a C++17 compiler before running Figures 2 and 4.')
        subprocess.run([compiler, '-O3', '-std=c++17', '-ffp-contract=off',
                        '-dynamiclib' if platform.system() == 'Darwin' else '-shared', '-fPIC',
                        str(source), '-o', str(temporary)], check=True)
        os.replace(temporary, destination)
    LIBRARY = ctypes.CDLL(str(destination))
    ptr = ctypes.POINTER(ctypes.c_float)
    LIBRARY.fluid_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr, ptr]
    LIBRARY.fluid_create.restype = ctypes.c_void_p
    LIBRARY.fluid_destroy.argtypes = [ctypes.c_void_p]
    LIBRARY.fluid_destroy.restype = None
    LIBRARY.fluid_evaluate.argtypes = [ctypes.c_void_p, ptr, ptr]
    LIBRARY.fluid_evaluate.restype = ctypes.c_float
    LIBRARY.fluid_density.argtypes = [ctypes.c_void_p, ptr]
    LIBRARY.fluid_density.restype = None
    return LIBRARY


def pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class FluidProblem:
    def __init__(self, steps=64, n=64, pressure=25):
        self.steps, self.n, self.pressure = steps, n, pressure
        self.initial_density = np.asarray(Image.open(ASSETS / 'init_image.png').convert('L').resize((n, n)), dtype=np.float32) / 255
        self.target_density = np.asarray(Image.open(ASSETS / 'target_image.png').convert('L').resize((n, n)), dtype=np.float32) / 255
        self.library = native_library()
        self.context = self.library.fluid_create(n, steps, pressure, pointer(self.initial_density), pointer(self.target_density))
        self._gradient_problem = None

    def objective(self, x):
        velocity = np.ascontiguousarray(x, dtype=np.float32)
        return float(self.library.fluid_evaluate(self.context, pointer(velocity), None))

    def current_density(self):
        density = np.empty((self.n, self.n), dtype=np.float32)
        self.library.fluid_density(self.context, pointer(density))
        return density

    def gradient(self, x):
        if self._gradient_problem is None:
            from .gradient import WarpGradient, FluidSettings
            self._gradient_problem = WarpGradient(sim_steps=self.steps, settings=FluidSettings(self.n, self.pressure),
                initial_image_path=ASSETS / 'init_image.png', target_image_path=ASSETS / 'target_image.png')
        parameters = np.asarray(x).reshape(2, self.n, self.n)
        return np.concatenate([g.ravel() for g in self._gradient_problem.gradient_at(parameters)])

    def projector(self, x):
        return lambda directions: directions

    def close(self):
        if self.context:
            self.library.fluid_destroy(self.context)
            self.context = None

    def __del__(self):
        if getattr(self, 'context', None):
            self.close()
