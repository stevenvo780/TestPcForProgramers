#!/usr/bin/env python3
"""
Simulación de N-cuerpos con Visualización en Tiempo Real
========================================================

Un proyecto científico computacionalmente intensivo que simula la interacción
gravitacional entre múltiples cuerpos celestes usando el algoritmo de fuerza bruta O(n²).

Este proyecto es ideal para estresar la CPU ya que:
- Calcula todas las fuerzas gravitacionales entre todos los pares de cuerpos (O(n²))
- Integra numéricamente las ecuaciones de movimiento usando Runge-Kutta
- Renderiza visualización en tiempo real
- Soporta hasta miles de cuerpos

Autor: Sistema de Simulación Científica
"""

import os
import shutil
import sys
import numpy as np
import matplotlib
# Permitir seleccionar backend por CLI muy temprano: --backend <NombreBackend>
_cli_backend = None
_viewer_cli = None
try:
    if '--backend' in sys.argv:
        idx = sys.argv.index('--backend')
        if idx + 1 < len(sys.argv):
            _cli_backend = sys.argv[idx + 1]
    if '--viewer' in sys.argv:
        idxv = sys.argv.index('--viewer')
        if idxv + 1 < len(sys.argv):
            _viewer_cli = sys.argv[idxv + 1]
except Exception:
    _cli_backend = _cli_backend or None
    _viewer_cli = _viewer_cli or None
# Selección de backend: forzar Agg para viewer web/opencv o en headless; si no, respetar CLI o elegir interactivo si hay DISPLAY
if (_viewer_cli in ('web', 'opencv')) or ('--headless' in sys.argv):
    try:
        matplotlib.use('Agg')
        if _viewer_cli:
            print(f"[Info] Usando backend 'Agg' para viewer '{_viewer_cli}'.")
        else:
            print("[Info] Usando backend 'Agg' en modo headless.")
    except Exception:
        pass
elif _cli_backend:
    try:
        matplotlib.use(_cli_backend)
        print(f"[Info] Backend forzado por CLI: {_cli_backend}")
    except Exception as e:
        print(f"[Aviso] No se pudo usar backend '{_cli_backend}': {e}")
elif not os.environ.get('DISPLAY'):
    # Sin DISPLAY: advertir y usar Agg (no habrá ventana). Puedes export DISPLAY o usar --headless.
    print("[Aviso] No se detecta DISPLAY. Usando backend 'Agg' (sin ventana). Para ver en vivo, ejecuta en un entorno con X/Wayland o exporta DISPLAY.")
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
else:
    # Tenemos DISPLAY y no se forzó backend: intentar QtAgg, luego TkAgg, luego dejar default
    for candidate in ('QtAgg', 'TkAgg'):
        try:
            matplotlib.use(candidate)
            print(f"[Info] Backend interactivo seleccionado: {candidate}")
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import argparse
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# Configurar NumPy para usar múltiples hilos en operaciones BLAS/LAPACK
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
os.environ['VECLIB_MAXIMUM_THREADS'] = str(mp.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(mp.cpu_count())

# Intentar importar numba para JIT compilation
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Decorador dummy si numba no está disponible
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    prange = range

# Constantes físicas
G = 6.67430e-11  # Constante gravitacional universal (m³/kg·s²)
AU = 1.496e11    # Unidad astronómica (m)
SOLAR_MASS = 1.989e30  # Masa del sol (kg)
EARTH_MASS = 5.972e24  # Masa de la Tierra (kg)

# Constantes de simulación (escaladas para mejor visualización)
G_SIM = 1.0      # Constante gravitacional escalada
SOFTENING = 0.1  # Factor de suavizado para evitar singularidades


@dataclass
class Body:
    """
    Representa un cuerpo celeste en la simulación.
    
    Attributes:
        mass: Masa del cuerpo (kg)
        x, y: Posición en coordenadas cartesianas (m)
        vx, vy: Velocidad en coordenadas cartesianas (m/s)
        fx, fy: Fuerza total aplicada al cuerpo (N)
        trail: Lista de posiciones históricas para visualización
        color: Color para visualización
        size: Tamaño para visualización
    """
    mass: float
    x: float
    y: float
    vx: float
    vy: float
    fx: float = 0.0
    fy: float = 0.0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    color: str = 'blue'
    size: float = 10.0
    
    def __post_init__(self):
        # Asegurar valores coherentes si se instanció con parámetros atípicos
        if self.trail is None:  # legacy compat
            self.trail = []
    
    def to_array(self):
        """Convierte el cuerpo a arrays NumPy para operaciones vectorizadas."""
        return np.array([self.mass, self.x, self.y, self.vx, self.vy, self.fx, self.fy])
    
    def from_array(self, array):
        """Actualiza el cuerpo desde arrays NumPy."""
        self.mass, self.x, self.y, self.vx, self.vy, self.fx, self.fy = array


# ============================================================================
# FUNCIONES OPTIMIZADAS CON NUMBA Y PARALELIZACIÓN
# ============================================================================

@njit(parallel=True, fastmath=True)
def calculate_forces_vectorized(positions, masses, forces, softening=0.1, g_sim=1.0):
    """
    Calcula fuerzas gravitacionales usando Numba JIT compilation con paralelización.
    
    Args:
        positions: Array (N, 2) con posiciones [x, y] de los cuerpos
        masses: Array (N,) con las masas de los cuerpos
        forces: Array (N, 2) para almacenar las fuerzas resultantes
        softening: Factor de suavizado
        g_sim: Constante gravitacional de simulación
    """
    n = positions.shape[0]
    
    # Resetear fuerzas
    for i in prange(n):
        forces[i, 0] = 0.0
        forces[i, 1] = 0.0
    
    # Cálculo paralelo de fuerzas entre todos los pares
    for i in prange(n):
        for j in range(i + 1, n):
            # Vector distancia
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            # Distancia con suavizado
            r_squared = dx*dx + dy*dy + softening*softening
            r = np.sqrt(r_squared)
            r_cubed = r_squared * r
            
            # Magnitud de la fuerza
            force_magnitude = g_sim * masses[i] * masses[j] / r_squared
            
            # Componentes de fuerza
            fx = force_magnitude * dx / r
            fy = force_magnitude * dy / r
            
            # Aplicar fuerzas (tercera ley de Newton)
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[j, 0] -= fx
            forces[j, 1] -= fy


@njit(fastmath=True)
def integrate_euler_vectorized(positions, velocities, forces, masses, dt):
    """
    Integración Euler vectorizada con Numba.
    
    Args:
        positions: Array (N, 2) con posiciones
        velocities: Array (N, 2) con velocidades
        forces: Array (N, 2) con fuerzas
        masses: Array (N,) con masas
        dt: Paso de tiempo
    """
    n = positions.shape[0]
    
    for i in range(n):
        # Actualizar velocidades: v = v + a*dt
        ax = forces[i, 0] / masses[i]
        ay = forces[i, 1] / masses[i]
        velocities[i, 0] += ax * dt
        velocities[i, 1] += ay * dt
        
        # Actualizar posiciones: x = x + v*dt
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt


def calculate_forces_chunk(args):
    """
    Calcula fuerzas para un chunk de cuerpos (para multiprocessing).
    
    Args:
        args: Tupla (start_idx, end_idx, positions, masses, softening, g_sim)
    
    Returns:
        Array con las fuerzas calculadas para el chunk
    """
    start_idx, end_idx, positions, masses, softening, g_sim = args
    n = positions.shape[0]
    chunk_size = end_idx - start_idx
    forces = np.zeros((chunk_size, 2))
    
    # Calcular fuerzas para cada cuerpo en el chunk
    for local_i, global_i in enumerate(range(start_idx, end_idx)):
        fx, fy = 0.0, 0.0
        
        # Interacciones con todos los otros cuerpos
        for j in range(n):
            if global_i == j:
                continue
                
            # Vector distancia
            dx = positions[j, 0] - positions[global_i, 0]
            dy = positions[j, 1] - positions[global_i, 1]
            
            # Distancia con suavizado
            r_squared = dx*dx + dy*dy + softening*softening
            r = np.sqrt(r_squared)
            
            # Magnitud de la fuerza
            force_magnitude = g_sim * masses[global_i] * masses[j] / r_squared
            
            # Componentes de fuerza
            fx += force_magnitude * dx / r
            fy += force_magnitude * dy / r
        
        forces[local_i, 0] = fx
        forces[local_i, 1] = fy
    
    return start_idx, end_idx, forces


class NBodySimulation:
    """
    Sistema principal de simulación N-cuerpos.
    
    Implementa el algoritmo clásico de fuerza bruta para calcular las
    interacciones gravitacionales entre todos los pares de cuerpos.
    
    La complejidad computacional es O(n²) por paso de tiempo, lo que
    hace que sea muy demandante para la CPU con muchos cuerpos.
    """
    
    def __init__(self, bodies: List[Body], dt: float = 0.01, 
                 use_rk4: bool = True, max_trail_length: int = 1000,
                 parallel_method: str = 'numba', num_processes: Optional[int] = None):
        """
        Inicializa la simulación.
        
        Args:
            bodies: Lista de cuerpos a simular
            dt: Paso de tiempo para la integración numérica
            use_rk4: Si usar Runge-Kutta 4 (más preciso) o Euler (más rápido)
            max_trail_length: Longitud máxima de las estelas
            parallel_method: Método de paralelización ('numba', 'multiprocessing', 'serial')
            num_processes: Número de procesos para multiprocessing (None = auto-detectar)
        """
        self.bodies = bodies
        self.dt = dt
        self.use_rk4 = use_rk4
        self.max_trail_length = max_trail_length
        self.parallel_method = parallel_method
        self.num_processes = num_processes or mp.cpu_count()
        
        # Validar método de paralelización
        valid_methods = ['numba', 'multiprocessing', 'serial']
        if parallel_method not in valid_methods:
            raise ValueError(f"parallel_method debe ser uno de: {valid_methods}")
            
        if parallel_method == 'numba' and not NUMBA_AVAILABLE:
            warnings.warn("Numba no está disponible, usando método serial", RuntimeWarning)
            self.parallel_method = 'serial'
        
        self.time = 0.0
        self.step_count = 0
        self.total_energy_history = []
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'fps': [],
            'computation_time': [],
            'parallel_efficiency': []
        }
        
        # Arrays NumPy para operaciones vectorizadas
        self._update_arrays()
        
        # Pool de procesos para multiprocessing (inicializar una vez)
        self._process_pool = None
        if self.parallel_method == 'multiprocessing':
            self._process_pool = ProcessPoolExecutor(max_workers=self.num_processes)
    
    def _update_arrays(self):
        """Actualiza los arrays NumPy desde la lista de cuerpos."""
        n = len(self.bodies)
        self.positions = np.zeros((n, 2))
        self.velocities = np.zeros((n, 2))
        self.forces = np.zeros((n, 2))
        self.masses = np.zeros(n)
        
        for i, body in enumerate(self.bodies):
            self.positions[i] = [body.x, body.y]
            self.velocities[i] = [body.vx, body.vy]
            self.forces[i] = [body.fx, body.fy]
            self.masses[i] = body.mass
    
    def _sync_from_arrays(self):
        """Sincroniza la lista de cuerpos desde los arrays NumPy."""
        for i, body in enumerate(self.bodies):
            body.x, body.y = self.positions[i]
            body.vx, body.vy = self.velocities[i]
            body.fx, body.fy = self.forces[i]
    
    def __del__(self):
        """Limpia el pool de procesos al destruir el objeto."""
        if hasattr(self, '_process_pool') and self._process_pool is not None:
            self._process_pool.shutdown(wait=False)
        
    def calculate_forces(self):
        """
        Calcula las fuerzas gravitacionales usando el método de paralelización seleccionado.
        
        Esta es la parte más computacionalmente intensiva: O(n²).
        Para n cuerpos, realiza n(n-1)/2 cálculos de fuerza.
        """
        parallel_start_time = time.time()
        
        # Actualizar arrays desde cuerpos
        self._update_arrays()
        
        if self.parallel_method == 'numba' and NUMBA_AVAILABLE:
            # Usar Numba JIT con paralelización automática
            calculate_forces_vectorized(
                self.positions, self.masses, self.forces, 
                SOFTENING, G_SIM
            )
        
        elif self.parallel_method == 'multiprocessing':
            # Usar multiprocessing para distribuir trabajo
            self._calculate_forces_multiprocessing()
        
        else:
            # Método serial (fallback)
            self._calculate_forces_serial()
        
        # Sincronizar de vuelta a los objetos Body
        self._sync_from_arrays()
        
        # Calcular eficiencia de paralelización
        parallel_time = time.time() - parallel_start_time
        if len(self.performance_metrics['computation_time']) > 0:
            # Comparar con tiempo promedio anterior
            avg_time = np.mean(self.performance_metrics['computation_time'][-10:])
            efficiency = avg_time / parallel_time if parallel_time > 0 else 1.0
            self.performance_metrics['parallel_efficiency'].append(min(float(efficiency), float(self.num_processes)))
    
    def _calculate_forces_serial(self):
        """Método serial original para comparación."""
        n = len(self.positions)
        self.forces.fill(0.0)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Vector distancia
                dx = self.positions[j, 0] - self.positions[i, 0]
                dy = self.positions[j, 1] - self.positions[i, 1]
                
                # Distancia con suavizado
                r_squared = dx*dx + dy*dy + SOFTENING*SOFTENING
                r = np.sqrt(r_squared)
                
                # Magnitud de la fuerza
                force_magnitude = G_SIM * self.masses[i] * self.masses[j] / r_squared
                
                # Componentes de fuerza
                fx = force_magnitude * dx / r
                fy = force_magnitude * dy / r
                
                # Aplicar fuerzas
                self.forces[i, 0] += fx
                self.forces[i, 1] += fy
                self.forces[j, 0] -= fx
                self.forces[j, 1] -= fy
    
    def _calculate_forces_multiprocessing(self):
        """Cálculo de fuerzas usando multiprocessing."""
        n = len(self.positions)
        self.forces.fill(0.0)
        # Asegurar que el pool exista (robustez)
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.num_processes)
        
        # Determinar tamaño de chunk óptimo
        chunk_size = max(1, n // (self.num_processes * 2))  # 2x procesos para mejor balance
        
        # Crear chunks de trabajo
        chunks = []
        for i in range(0, n, chunk_size):
            end_idx = min(i + chunk_size, n)
            chunks.append((i, end_idx, self.positions, self.masses, SOFTENING, G_SIM))
        
        if len(chunks) == 1:
            # Si solo hay un chunk, usar método serial
            self._calculate_forces_serial()
            return
        
        # Procesar chunks en paralelo
        future_to_chunk = {}
        for chunk_args in chunks:
            future = self._process_pool.submit(calculate_forces_chunk, chunk_args)
            future_to_chunk[future] = chunk_args[0:2]  # start_idx, end_idx
        
        # Recolectar resultados
        for future in as_completed(future_to_chunk):
            start_idx, end_idx, chunk_forces = future.result()
            self.forces[start_idx:end_idx] = chunk_forces
    
    def integrate_euler(self):
        """Integración numérica usando el método de Euler (primera orden) vectorizado."""
        if self.parallel_method in ['numba', 'multiprocessing']:
            # Usar versión vectorizada
            integrate_euler_vectorized(
                self.positions, self.velocities, self.forces, self.masses, self.dt
            )
            self._sync_from_arrays()
        else:
            # Método serial original
            for body in self.bodies:
                # Actualizar velocidades: v = v + a*dt
                ax = body.fx / body.mass
                ay = body.fy / body.mass
                body.vx += ax * self.dt
                body.vy += ay * self.dt
                
                # Actualizar posiciones: x = x + v*dt
                body.x += body.vx * self.dt
                body.y += body.vy * self.dt
    
    def integrate_rk4(self):
        """
        Integración numérica usando Runge-Kutta de 4to orden.
        
        Mucho más preciso que Euler, pero requiere 4x más cálculos.
        Ideal para estresar la CPU manteniendo precisión numérica.
        """
        dt = self.dt
        
        # Almacenar estado inicial
        initial_state = []
        for body in self.bodies:
            initial_state.append((body.x, body.y, body.vx, body.vy))
        
        # Función para evaluar derivadas
        def derivatives(bodies_state):
            # Restaurar estado temporal
            for i, (x, y, vx, vy) in enumerate(bodies_state):
                self.bodies[i].x = x
                self.bodies[i].y = y
                self.bodies[i].vx = vx
                self.bodies[i].vy = vy
            
            # Calcular fuerzas
            self.calculate_forces()
            
            # Retornar derivadas [dx/dt, dy/dt, dvx/dt, dvy/dt]
            derivs = []
            for body in self.bodies:
                ax = body.fx / body.mass
                ay = body.fy / body.mass
                derivs.append((body.vx, body.vy, ax, ay))
            return derivs
        
        # RK4: k1, k2, k3, k4
        k1 = derivatives(initial_state)
        
        k1_state = []
        for i, (x, y, vx, vy) in enumerate(initial_state):
            dx1, dy1, dvx1, dvy1 = k1[i]
            k1_state.append((x + 0.5*dt*dx1, y + 0.5*dt*dy1, 
                           vx + 0.5*dt*dvx1, vy + 0.5*dt*dvy1))
        k2 = derivatives(k1_state)
        
        k2_state = []
        for i, (x, y, vx, vy) in enumerate(initial_state):
            dx2, dy2, dvx2, dvy2 = k2[i]
            k2_state.append((x + 0.5*dt*dx2, y + 0.5*dt*dy2,
                           vx + 0.5*dt*dvx2, vy + 0.5*dt*dvy2))
        k3 = derivatives(k2_state)
        
        k3_state = []
        for i, (x, y, vx, vy) in enumerate(initial_state):
            dx3, dy3, dvx3, dvy3 = k3[i]
            k3_state.append((x + dt*dx3, y + dt*dy3,
                           vx + dt*dvx3, vy + dt*dvy3))
        k4 = derivatives(k3_state)
        
        # Actualizar usando la fórmula RK4
        for i, (x, y, vx, vy) in enumerate(initial_state):
            dx1, dy1, dvx1, dvy1 = k1[i]
            dx2, dy2, dvx2, dvy2 = k2[i]
            dx3, dy3, dvx3, dvy3 = k3[i]
            dx4, dy4, dvx4, dvy4 = k4[i]
            
            self.bodies[i].x = x + dt/6 * (dx1 + 2*dx2 + 2*dx3 + dx4)
            self.bodies[i].y = y + dt/6 * (dy1 + 2*dy2 + 2*dy3 + dy4)
            self.bodies[i].vx = vx + dt/6 * (dvx1 + 2*dvx2 + 2*dvx3 + dvx4)
            self.bodies[i].vy = vy + dt/6 * (dvy1 + 2*dvy2 + 2*dvy3 + dvy4)
    
    def calculate_total_energy(self) -> float:
        """
        Calcula la energía total del sistema (cinética + potencial).
        
        La conservación de energía es una buena métrica para validar
        la precisión numérica de la simulación.
        """
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        # Energía cinética: KE = 1/2 * m * v²
        for body in self.bodies:
            v_squared = body.vx*body.vx + body.vy*body.vy
            kinetic_energy += 0.5 * body.mass * v_squared
        
        # Energía potencial gravitacional: PE = -G*m1*m2/r
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                body1, body2 = self.bodies[i], self.bodies[j]
                dx = body2.x - body1.x
                dy = body2.y - body1.y
                r = np.sqrt(dx*dx + dy*dy + SOFTENING*SOFTENING)
                potential_energy -= G_SIM * body1.mass * body2.mass / r
        
        return kinetic_energy + potential_energy
    
    def update_trails(self):
        """Actualiza las estelas de los cuerpos para visualización."""
        for body in self.bodies:
            body.trail.append((body.x, body.y))
            if len(body.trail) > self.max_trail_length:
                body.trail.pop(0)
    
    def step(self):
        """
        Ejecuta un paso completo de la simulación.
        
        Este es el núcleo computacional que se ejecuta en cada frame:
        1. Calcula todas las fuerzas O(n²)
        2. Integra las ecuaciones de movimiento
        3. Actualiza métricas y estelas
        """
        start_time = time.time()
        
        # Calcular fuerzas (la parte más intensiva computacionalmente)
        self.calculate_forces()
        
        # Integrar ecuaciones de movimiento
        if self.use_rk4:
            self.integrate_rk4()
        else:
            self.integrate_euler()
        
        # Actualizar estado de la simulación
        self.time += self.dt
        self.step_count += 1
        self.update_trails()
        
        # Calcular métricas de rendimiento
        computation_time = time.time() - start_time
        self.performance_metrics['computation_time'].append(computation_time)
        
        # Conservar solo las últimas 1000 métricas para evitar consumo excesivo de memoria
        if len(self.performance_metrics['computation_time']) > 1000:
            for key in self.performance_metrics:
                if self.performance_metrics[key]:
                    self.performance_metrics[key].pop(0)


class Visualizer:
    """
    Sistema de visualización en tiempo real para la simulación N-cuerpos.
    
    Crea una animación matplotlib que muestra:
    - Posiciones actuales de los cuerpos
    - Estelas de movimiento
    - Métricas de rendimiento en tiempo real
    - Información del sistema
    """
    
    def __init__(self, simulation: NBodySimulation, window_size: float = 10.0,
                 show_trails: bool = True, show_metrics: bool = True,
                 color_mode: str = 'fixed', cmap: str = 'viridis',
                 show_vectors: bool = False, show_density: bool = False,
                 use_blit: bool = False):
        self.simulation = simulation
        self._record_live = False
        self._writer = None
        self._writer_ctx = None
        self.window_size = window_size
        self.show_trails = show_trails
        self.show_metrics = show_metrics
        self.color_mode = color_mode
        self.cmap = plt.get_cmap(cmap)
        self.show_vectors = show_vectors
        self.show_density = show_density
        self.use_blit = use_blit
        
        # Configurar la figura y subplots
        if show_metrics:
            self.fig, (self.ax_sim, self.ax_metrics) = plt.subplots(1, 2, figsize=(16, 8))
            self.ax_metrics.set_title('Métricas de Rendimiento')
        else:
            self.fig, self.ax_sim = plt.subplots(figsize=(12, 8))
        
        self.ax_sim.set_xlim(-window_size, window_size)
        self.ax_sim.set_ylim(-window_size, window_size)
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_title('Simulación N-Cuerpos en Tiempo Real')
        self.ax_sim.set_xlabel('Posición X')
        self.ax_sim.set_ylabel('Posición Y')
        self.ax_sim.grid(True, alpha=0.3)
        
        # Elementos gráficos
        self.body_scatter = None
        self.trail_lines = []
        self.metrics_lines = []
        self.quiver = None
        
        # Para el monitoreo de rendimiento
        self.last_time = time.time()
        self.frame_count = 0
        
    def animate(self, frame):
        """Función de animación llamada por matplotlib en cada frame. Retorna artistas para blitting."""
        current_time = time.time()
        artists = []

        # Ejecutar paso de simulación
        self.simulation.step()

        # Actualizar visualización
        self.ax_sim.clear()
        self.ax_sim.set_xlim(-self.window_size, self.window_size)
        self.ax_sim.set_ylim(-self.window_size, self.window_size)
        self.ax_sim.set_aspect('equal')
        self.ax_sim.grid(True, alpha=0.3)

        # Dibujar estelas
        if self.show_trails:
            for body in self.simulation.bodies:
                if len(body.trail) > 1:
                    trail_x, trail_y = zip(*body.trail)
                    alpha_values = np.linspace(0.1, 0.8, len(trail_x))
                    for i in range(len(trail_x) - 1):
                        lines = self.ax_sim.plot(
                            [trail_x[i], trail_x[i + 1]],
                            [trail_y[i], trail_y[i + 1]],
                            color=body.color,
                            alpha=alpha_values[i],
                            linewidth=1,
                        )
                        artists.extend(lines)

        # Datos base
        x_positions = np.array([body.x for body in self.simulation.bodies])
        y_positions = np.array([body.y for body in self.simulation.bodies])
        vxs = np.array([body.vx for body in self.simulation.bodies])
        vys = np.array([body.vy for body in self.simulation.bodies])
        masses = np.array([body.mass for body in self.simulation.bodies])
        speeds = np.sqrt(vxs*vxs + vys*vys)
        sizes = [body.size * np.log10(body.mass + 1) for body in self.simulation.bodies]

        # Colores según modo
        if self.color_mode == 'fixed':
            colors = [body.color for body in self.simulation.bodies]
        elif self.color_mode == 'speed':
            # Normalizar velocidades para colormap
            smin, smax = np.percentile(speeds, [5, 95])
            smax = max(smax, smin + 1e-6)
            norm_s = np.clip((speeds - smin) / (smax - smin), 0, 1)
            colors = self.cmap(norm_s)
        else:  # mass
            mmin, mmax = np.percentile(masses, [5, 95])
            mmax = max(mmax, mmin + 1e-6)
            norm_m = np.clip((masses - mmin) / (mmax - mmin), 0, 1)
            colors = self.cmap(norm_m)

        # Densidad (hexbin) bajo los puntos
        if self.show_density:
            hb = self.ax_sim.hexbin(x_positions, y_positions, gridsize=35, cmap='inferno', mincnt=1, alpha=0.6)
            artists.append(hb)

        # Puntos
        sc = self.ax_sim.scatter(x_positions, y_positions, c=colors, s=sizes, alpha=0.85, edgecolors='black')
        artists.append(sc)

        # Vectores de velocidad
        if self.show_vectors:
            scale = max(np.max(speeds), 1e-6)
            self.quiver = self.ax_sim.quiver(x_positions, y_positions, vxs/scale, vys/scale, angles='xy', scale_units='xy', scale=0.2, color='white', alpha=0.7)
            artists.append(self.quiver)

        # Título con información
        n_bodies = len(self.simulation.bodies)
        self.ax_sim.set_title(
            f'Simulación N-Cuerpos: {n_bodies} cuerpos | '
            f'Tiempo: {self.simulation.time:.1f}s | '
            f'Paso: {self.simulation.step_count}'
        )

        # Actualizar métricas de rendimiento
        if frame % 10 == 0:  # Actualizar cada 10 frames para reducir overhead
            self.update_performance_metrics(current_time)

        # Mostrar métricas si está habilitado
        if self.show_metrics and frame % 30 == 0:  # Actualizar cada 30 frames
            self.update_metrics_plot()

        self.last_time = current_time
        self.frame_count += 1
        # Si la grabación en vivo está activa, capturar el frame actual
        if self._record_live and self._writer is not None:
            try:
                self._writer.grab_frame()
            except Exception:
                # No interrumpir la animación si falla la captura
                pass
        return artists
    
    def update_performance_metrics(self, current_time):
        """Actualiza las métricas de rendimiento del sistema."""
        # CPU y memoria
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.simulation.performance_metrics['cpu_usage'].append(cpu_percent)
        self.simulation.performance_metrics['memory_usage'].append(memory_percent)
        
        # FPS
        if self.frame_count > 0:
            fps = 1.0 / (current_time - self.last_time) if current_time != self.last_time else 0
            self.simulation.performance_metrics['fps'].append(fps)
    
    def update_metrics_plot(self):
        """Actualiza el subplot de métricas."""
        if not self.show_metrics:
            return
            
        self.ax_metrics.clear()
        
        metrics = self.simulation.performance_metrics
        
        if metrics['cpu_usage']:
            times = range(len(metrics['cpu_usage']))
            self.ax_metrics.plot(times, metrics['cpu_usage'][-100:], 
                               label='CPU %', color='red', alpha=0.7)
        
        if metrics['memory_usage']:
            times = range(len(metrics['memory_usage']))
            self.ax_metrics.plot(times, metrics['memory_usage'][-100:], 
                               label='Memory %', color='blue', alpha=0.7)
        
        if metrics['fps']:
            times = range(len(metrics['fps']))
            fps_scaled = [fps * 10 for fps in metrics['fps'][-100:]]  # Escalar para visibilidad
            self.ax_metrics.plot(times, fps_scaled, 
                               label='FPS x10', color='green', alpha=0.7)
        
        self.ax_metrics.set_title('Métricas de Rendimiento')
        self.ax_metrics.set_xlabel('Tiempo (frames)')
        self.ax_metrics.set_ylabel('Porcentaje / Valor')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)
        
        # Mostrar estadísticas en texto
        if metrics['computation_time']:
            avg_comp_time = np.mean(metrics['computation_time'][-100:]) * 1000  # ms
            self.ax_metrics.text(0.02, 0.98, f'Tiempo cálculo promedio: {avg_comp_time:.2f}ms', 
                               transform=self.ax_metrics.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _on_close(self, event=None):
        """Finaliza la grabación en vivo si está activa."""
        if self._record_live and self._writer_ctx is not None:
            try:
                # Cerrar contexto de escritura sin error
                self._writer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._record_live = False
    
    def start(self, interval=50, save_animation=False, filename='nbody_simulation.mp4',
              headless: bool = False, fps: int = 20, frames: Optional[int] = None,
              duration: Optional[float] = None, viewer: str = 'matplotlib'):
        """
        Inicia la visualización en tiempo real.
        
        Args:
            interval: Tiempo entre frames en millisegundos
            save_animation: Si guardar la animación como video
            filename: Nombre del archivo de video
            headless: Ejecutar sin mostrar ventana (no llama a plt.show())
            fps: Cuadros por segundo del video de salida
            frames: Número total de frames a generar (si None y duration no provisto, animación infinita)
            duration: Duración en segundos para derivar frames si frames no se especifica
        """
        print(f"Iniciando simulación con {len(self.simulation.bodies)} cuerpos...")
        print(f"Método de integración: {'Runge-Kutta 4' if self.simulation.use_rk4 else 'Euler'}")
        print(f"Paso de tiempo: {self.simulation.dt}")
        if frames is not None:
            print(f"Animación limitada a {frames} frames.")
        elif duration is not None:
            print(f"Animación limitada a {duration} segundos.")
        else:
            print("Presiona Ctrl+C para detener la simulación.")

        # Derivar frames desde duration si aplica
        if frames is None and duration is not None and fps > 0:
            frames = max(1, int(duration * fps))

        # Si se pasa fps, ajustar interval automáticamente si coincide
        if fps and interval == 50:
            interval = int(1000 / fps)

        if viewer == 'matplotlib':
            self.animation = animation.FuncAnimation(
                self.fig, self.animate, interval=interval, blit=False, cache_frame_data=False,
                frames=frames)
        elif viewer == 'opencv':
            if not OPENCV_AVAILABLE:
                raise RuntimeError("OpenCV no está disponible. Instala opencv-python.")
            # Bucle manual de renderizado con OpenCV
            print("[Info] Visor OpenCV activado (ventana en vivo)")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if save_animation else None
            writer = None
            if save_animation:
                writer = cv2.VideoWriter(filename, fourcc, fps, (800, 800))
                if not writer.isOpened():
                    writer = None
                    print("[Aviso] No se pudo abrir VideoWriter de OpenCV; se omitirá el guardado.")
            # Definir num frames
            total_frames = frames or (int(duration * fps) if duration else 0)
            i = 0
            try:
                while True:
                    self.simulation.step()
                    # Render a imagen usando matplotlib backend offscreen
                    self.ax_sim.clear()
                    self.ax_sim.set_xlim(-self.window_size, self.window_size)
                    self.ax_sim.set_ylim(-self.window_size, self.window_size)
                    self.ax_sim.set_aspect('equal')
                    self.ax_sim.grid(True, alpha=0.3)
                    xs = [b.x for b in self.simulation.bodies]
                    ys = [b.y for b in self.simulation.bodies]
                    cs = [b.color for b in self.simulation.bodies]
                    sz = [b.size * np.log10(b.mass + 1) for b in self.simulation.bodies]
                    self.ax_sim.scatter(xs, ys, c=cs, s=sz, alpha=0.8, edgecolors='black')
                    self.fig.canvas.draw()
                    # Convertir canvas a imagen usando buffer_rgba
                    w, h = self.fig.canvas.get_width_height()
                    buf = self.fig.canvas.buffer_rgba()
                    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
                    # Convertir RGBA -> BGR para OpenCV
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    # Redimensionar a 800x800 para ventana
                    img_resized = cv2.resize(img, (800, 800))
                    cv2.imshow('N-Body (OpenCV)', img_resized)
                    if writer is not None:
                        writer.write(img_resized)
                    # Tecla ESC para salir
                    if cv2.waitKey(max(1, int(interval))) & 0xFF == 27:
                        break
                    i += 1
                    if total_frames and i >= total_frames:
                        break
            finally:
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
            return
        elif viewer == 'web':
            # Visor web (Flask) con streaming MJPEG. No requiere backend interactivo.
            try:
                from flask import Flask, Response
            except Exception as e:
                raise RuntimeError("El visor web requiere Flask. Instala con: pip install Flask") from e

            if not OPENCV_AVAILABLE:
                raise RuntimeError("El visor web requiere OpenCV para codificar JPEG. Instala opencv-python.")

            app = Flask(__name__)
            stop_event = threading.Event()
            frame_lock = threading.Lock()
            last_frame = {'bytes': None}

            @app.route('/')
            def index():
                html = (
                    "<html><head><title>N-Body Web Viewer</title></head>"
                    "<body style='background:#111;color:#ddd;font-family:sans-serif;'>"
                    "<h2>N-Body Web Viewer (stream en vivo)</h2>"
                    "<p>FPS objetivo: %d</p>" % fps +
                    "<img src='/stream' style='max-width:96vw;border:1px solid #444;'/>"
                    "</body></html>"
                )
                return html

            def mjpeg_generator():
                boundary = b'--frame\r\n'
                while not stop_event.is_set():
                    with frame_lock:
                        data = last_frame['bytes']
                    if data is None:
                        # Frame placeholder negro 640x480 si aún no hay datos
                        black = np.zeros((480, 640, 3), dtype=np.uint8)
                        ok, enc = cv2.imencode('.jpg', black)
                        data = enc.tobytes() if ok else black.tobytes()
                    yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
                    time.sleep(max(0.001, 1.0/float(fps)))

            @app.route('/stream')
            def stream():
                return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

            def run_server():
                # Ejecutar servidor en hilo daemon
                app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

            # Preparar writer opcional (grabación mientras se transmite)
            live_writer = None
            live_ctx = None
            if save_animation:
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path is not None:
                    try:
                        live_writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='NBodySimulation'), bitrate=4000)
                    except Exception:
                        live_writer = None
                if live_writer is None:
                    try:
                        from matplotlib.animation import PillowWriter
                        if not filename.lower().endswith('.gif'):
                            base, _ = os.path.splitext(filename)
                            filename = base + '.gif'
                        live_writer = PillowWriter(fps=fps)
                    except Exception:
                        print("[Aviso] No se pudo crear writer. No se guardará video durante el stream.")

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            print("[Info] Visor web activo en http://127.0.0.1:5000 (abre en tu navegador)")

            total_frames = frames or (int(duration * fps) if duration else 0)
            produced = 0

            try:
                if save_animation and live_writer is not None:
                    live_ctx = live_writer.saving(self.fig, filename, dpi=100)
                    live_ctx.__enter__()

                # Bucle de producción de frames en el hilo principal
                while not stop_event.is_set():
                    self.simulation.step()
                    # Render
                    self.ax_sim.clear()
                    self.ax_sim.set_xlim(-self.window_size, self.window_size)
                    self.ax_sim.set_ylim(-self.window_size, self.window_size)
                    self.ax_sim.set_aspect('equal')
                    self.ax_sim.grid(True, alpha=0.3)
                    xs = [b.x for b in self.simulation.bodies]
                    ys = [b.y for b in self.simulation.bodies]
                    cs = [b.color for b in self.simulation.bodies]
                    sz = [b.size * np.log10(b.mass + 1) for b in self.simulation.bodies]
                    self.ax_sim.scatter(xs, ys, c=cs, s=sz, alpha=0.85, edgecolors='black')
                    self.fig.canvas.draw()
                    # Canvas a imagen (RGBA)
                    w, h = self.fig.canvas.get_width_height()
                    buf = self.fig.canvas.buffer_rgba()
                    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
                    # Convertir RGBA->BGR
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    # Codificar a JPEG
                    ok, enc = cv2.imencode('.jpg', bgr)
                    if ok:
                        with frame_lock:
                            last_frame['bytes'] = enc.tobytes()

                    if save_animation and live_writer is not None:
                        try:
                            live_writer.grab_frame()
                        except Exception:
                            pass

                    produced += 1
                    if total_frames and produced >= total_frames:
                        break

                    # Regular FPS aproximado
                    time.sleep(max(0.001, 1.0/float(fps)))
            finally:
                stop_event.set()
                if live_ctx is not None:
                    try:
                        live_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
            print("[Info] Streaming web finalizado.")
            return
        else:
            raise ValueError("viewer debe ser 'matplotlib', 'opencv' o 'web'")

        # Guardar animación: dos modos
        # - headless True: render offline a archivo
        # - headless False: grabación en vivo mientras se muestra la ventana
        if save_animation and headless:
            print(f"Renderizando (offline) animación como {filename}...")
            writer = None
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path is not None:
                try:
                    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='NBodySimulation'), bitrate=4000)
                except Exception as e:
                    print(f"No se pudo inicializar FFMpegWriter ({e}). Se intentará con PillowWriter (GIF).")
            if writer is None:
                try:
                    from matplotlib.animation import PillowWriter
                    if not filename.lower().endswith('.gif'):
                        base, _ = os.path.splitext(filename)
                        filename = base + '.gif'
                    writer = PillowWriter(fps=fps)
                except Exception as e:
                    raise RuntimeError("No se pudo crear un writer para la animación. Instala ffmpeg o pillow.") from e
            self.animation.save(filename, writer=writer)
            print(f"Animación guardada en: {os.path.abspath(filename)}")
        elif save_animation and not headless:
            # Grabación en vivo mientras se muestra la ventana
            print(f"Grabación en vivo activada, archivo: {filename}")
            writer = None
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path is not None:
                try:
                    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='NBodySimulation'), bitrate=4000)
                except Exception as e:
                    print(f"No se pudo inicializar FFMpegWriter ({e}). Se intentará con PillowWriter (GIF).")
            if writer is None:
                try:
                    from matplotlib.animation import PillowWriter
                    if not filename.lower().endswith('.gif'):
                        base, _ = os.path.splitext(filename)
                        filename = base + '.gif'
                    writer = PillowWriter(fps=fps)
                except Exception as e:
                    raise RuntimeError("No se pudo crear un writer para la animación. Instala ffmpeg o pillow.") from e
            # Iniciar contexto de guardado persistente y capturar frames en animate()
            self._writer = writer
            self._writer_ctx = writer.saving(self.fig, filename, dpi=100)
            try:
                self._writer_ctx.__enter__()
                self._record_live = True
            except Exception as e:
                print(f"No se pudo iniciar la grabación en vivo: {e}")
                self._record_live = False
            # Conectar cierre para finalizar writer
            self.fig.canvas.mpl_connect('close_event', self._on_close)

        plt.tight_layout()
        # Mostrar ventana cuando no es headless y el backend lo permite
        if not headless and viewer == 'matplotlib':
            backend = matplotlib.get_backend().lower()
            if backend == 'agg':
                print("[Aviso] Backend 'Agg' activo: no se puede abrir ventana. Guardando offline si --save fue pasado.")
                if save_animation and not headless:
                    # Si el usuario esperaba grabación en vivo pero no hay backend interactivo, ya hicimos offline arriba
                    pass
            else:
                # Autocierre: si se proporcionó duration o frames, cerrar la ventana al terminar
                if duration is not None:
                    ms = max(1, int(duration * 1000) + 500)
                    try:
                        timer = self.fig.canvas.new_timer(interval=ms)
                        timer.add_callback(lambda: plt.close(self.fig))
                        timer.start()
                        print(f"[Info] La ventana se cerrará automáticamente en ~{duration}s")
                    except Exception:
                        pass
                plt.show()
                # Al cerrar la ventana, asegurar cierre de writer si sigue activo
                self._on_close()


def create_solar_system() -> List[Body]:
    """
    Crea un sistema solar simplificado.
    
    Returns:
        Lista de cuerpos representando Sol, planetas interiores y algunos asteroides
    """
    bodies = []
    
    # Sol (en el centro)
    bodies.append(Body(
        mass=100.0, x=0.0, y=0.0, vx=0.0, vy=0.0,
        color='yellow', size=50.0
    ))
    
    # Planetas con órbitas aproximadamente circulares
    planet_data = [
        # masa, distancia, velocidad_orbital, color, tamaño
        (0.5, 2.0, 2.2, 'gray', 15),      # Mercurio
        (0.8, 3.0, 1.8, 'orange', 18),   # Venus
        (1.0, 4.0, 1.6, 'blue', 20),     # Tierra
        (0.6, 5.5, 1.3, 'red', 16),      # Marte
    ]
    
    for mass, distance, orbital_v, color, size in planet_data:
        bodies.append(Body(
            mass=mass, x=distance, y=0.0, vx=0.0, vy=orbital_v,
            color=color, size=size
        ))
    
    # Cinturón de asteroides
    np.random.seed(42)  # Para reproducibilidad
    for i in range(20):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(6.0, 8.0)
        mass = np.random.uniform(0.01, 0.05)
        orbital_v = np.sqrt(100.0 / distance) * 0.9  # Velocidad orbital aproximada
        
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        vx = -orbital_v * np.sin(angle)
        vy = orbital_v * np.cos(angle)
        
        bodies.append(Body(
            mass=mass, x=x, y=y, vx=vx, vy=vy,
            color='brown', size=8
        ))
    
    return bodies


def create_galaxy_cluster(n_bodies: int = 200) -> List[Body]:
    """
    Crea un cluster galáctico con múltiples centros de masa.
    
    Este escenario es extremadamente demandante computacionalmente
    debido al gran número de cuerpos (O(n²) interactions).
    
    Args:
        n_bodies: Número total de cuerpos a generar
    
    Returns:
        Lista de cuerpos representando un cluster galáctico
    """
    bodies = []
    np.random.seed(42)
    
    # Crear varios centros de masa (agujeros negros supermasivos)
    n_centers = max(3, n_bodies // 50)
    for i in range(n_centers):
        angle = 2 * np.pi * i / n_centers
        distance = np.random.uniform(8.0, 12.0)
        mass = np.random.uniform(50.0, 100.0)
        
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        
        # Velocidad para órbita aproximada alrededor del centro
        orbital_v = np.sqrt(500.0 / distance) * 0.5
        vx = -orbital_v * np.sin(angle)
        vy = orbital_v * np.cos(angle)
        
        bodies.append(Body(
            mass=mass, x=x, y=y, vx=vx, vy=vy,
            color='purple', size=40
        ))
    
    # Distribuir estrellas alrededor de los centros
    remaining_bodies = n_bodies - n_centers
    
    for i in range(remaining_bodies):
        # Elegir un centro aleatorio para orbitar
        center_idx = np.random.randint(n_centers)
        center = bodies[center_idx]
        
        # Posición aleatoria alrededor del centro
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.exponential(2.0) + 0.5  # Distribución exponencial
        
        x = center.x + distance * np.cos(angle)
        y = center.y + distance * np.sin(angle)
        
        # Masa y velocidad basadas en el tipo de estrella
        if np.random.random() < 0.1:  # 10% estrellas masivas
            mass = np.random.uniform(5.0, 15.0)
            color = 'white'
            size = 25
        elif np.random.random() < 0.3:  # 30% estrellas medianas  
            mass = np.random.uniform(1.0, 5.0)
            color = 'yellow'
            size = 15
        else:  # 60% enanas rojas
            mass = np.random.uniform(0.1, 1.0)
            color = 'red'
            size = 10
        
        # Velocidad orbital aproximada
        total_mass = center.mass
        orbital_v = np.sqrt(G_SIM * total_mass / max(distance, 0.1)) * np.random.uniform(0.8, 1.2)
        vx = center.vx - orbital_v * np.sin(angle)
        vy = center.vy + orbital_v * np.cos(angle)
        
        # Agregar perturbación aleatoria
        vx += np.random.normal(0, 0.3)
        vy += np.random.normal(0, 0.3)
        
        bodies.append(Body(
            mass=mass, x=x, y=y, vx=vx, vy=vy,
            color=color, size=size
        ))
    
    return bodies


def create_galaxy_collision(n_bodies_per_galaxy: int = 400, separation: float = 20.0,
                            relative_velocity: float = 1.0) -> List[Body]:
    """
    Escenario: colisión frontal de dos galaxias espirales simplificadas.

    - Dos discos de estrellas con bulbo central masivo
    - Trayectorias de colisión con leve desplazamiento para generar brazos de marea
    - Visualmente atractivo en tiempo real
    """
    np.random.seed(7)
    bodies: List[Body] = []

    def make_galaxy(cx: float, cy: float, vx_c: float, vy_c: float,
                     color_main: str, color_alt: str) -> List[Body]:
        g_bodies: List[Body] = []
        # Bulbo central
        g_bodies.append(Body(
            mass=150.0, x=cx, y=cy, vx=vx_c, vy=vy_c, color='white', size=45
        ))
        # Disco: distribución radial ~ exponencial y rotación diferencial
        for i in range(n_bodies_per_galaxy):
            r = np.random.exponential(4.0) + 0.2  # radio
            ang = np.random.uniform(0, 2*np.pi)
            x = cx + r * np.cos(ang)
            y = cy + r * np.sin(ang)
            # velocidad de rotación aprox v ~ sqrt(M/r)
            v = np.sqrt(G_SIM * 200.0 / max(r, 0.2)) * np.random.uniform(0.9, 1.1)
            # dirección tangencial + velocidad del centro
            vx = vx_c - v * np.sin(ang)
            vy = vy_c + v * np.cos(ang)
            mass = np.random.uniform(0.5, 2.0)
            col = color_main if i % 2 == 0 else color_alt
            size = np.clip(10 + np.random.normal(0, 3), 5, 20)
            g_bodies.append(Body(mass=mass, x=x, y=y, vx=vx, vy=vy, color=col, size=size))
        return g_bodies

    # Galaxia A
    bodies += make_galaxy(-separation/2, 0.0, +relative_velocity, 0.2,
                          color_main='cyan', color_alt='blue')
    # Galaxia B
    bodies += make_galaxy(+separation/2, 1.5, -relative_velocity, -0.2,
                          color_main='orange', color_alt='red')

    return bodies

# (Se mantiene una única definición de create_galaxy_collision más arriba)

def create_benchmark_scenario(n_bodies: int = 1000) -> List[Body]:
    """
    Crea un escenario específicamente diseñado para benchmark de CPU.
    
    Genera una distribución uniforme de cuerpos con masas similares
    para maximizar la carga computacional (no hay simplificaciones posibles).
    
    Args:
        n_bodies: Número de cuerpos (¡cuidado con números altos!)
    
    Returns:
        Lista de cuerpos para benchmark intensivo
    """
    bodies = []
    np.random.seed(42)
    
    print(f"ADVERTENCIA: Creando escenario de benchmark con {n_bodies} cuerpos.")
    print(f"Esto requerirá {n_bodies * (n_bodies - 1) // 2:,} cálculos de fuerza por paso.")
    print("¡Esto será MUY intensivo para la CPU!")
    
    # Distribución uniforme en un cubo
    size = 20.0
    for i in range(n_bodies):
        # Posición aleatoria
        x = np.random.uniform(-size, size)
        y = np.random.uniform(-size, size)
        
        # Velocidad aleatoria pequeña
        vx = np.random.uniform(-1.0, 1.0)
        vy = np.random.uniform(-1.0, 1.0)
        
        # Masa similar para todos (no permite optimizaciones)
        mass = np.random.uniform(0.8, 1.2)
        
        # Color basado en la posición para visualización
        if abs(x) > abs(y):
            color = 'red' if x > 0 else 'blue'
        else:
            color = 'green' if y > 0 else 'orange'
        
        bodies.append(Body(
            mass=mass, x=x, y=y, vx=vx, vy=vy,
            color=color, size=8
        ))
    
    return bodies


def print_performance_summary(simulation: NBodySimulation):
    """Imprime un resumen de rendimiento de la simulación."""
    metrics = simulation.performance_metrics
    
    if not metrics['computation_time']:
        print("No hay datos de rendimiento disponibles.")
        return
    
    print("\n" + "="*60)
    print("RESUMEN DE RENDIMIENTO")
    print("="*60)
    
    n_bodies = len(simulation.bodies)
    total_interactions = n_bodies * (n_bodies - 1) // 2
    
    print(f"Número de cuerpos: {n_bodies}")
    print(f"Interacciones por paso: {total_interactions:,}")
    print(f"Método de integración: {'Runge-Kutta 4' if simulation.use_rk4 else 'Euler'}")
    print(f"Paso de tiempo: {simulation.dt}")
    print(f"Tiempo simulado: {simulation.time:.2f}s")
    print(f"Pasos ejecutados: {simulation.step_count:,}")
    
    # Estadísticas de tiempo de cálculo
    comp_times = np.array(metrics['computation_time']) * 1000  # Convertir a ms
    print(f"\nTiempo de cálculo por paso:")
    print(f"  Promedio: {np.mean(comp_times):.2f}ms")
    print(f"  Mínimo: {np.min(comp_times):.2f}ms") 
    print(f"  Máximo: {np.max(comp_times):.2f}ms")
    print(f"  Desviación estándar: {np.std(comp_times):.2f}ms")
    
    # Rendimiento computacional
    if len(comp_times) > 0:
        avg_time_per_interaction = np.mean(comp_times) / total_interactions if total_interactions > 0 else 0
        interactions_per_second = total_interactions / (np.mean(comp_times) / 1000) if np.mean(comp_times) > 0 else 0
        
        print(f"\nRendimiento computacional:")
        print(f"  Tiempo por interacción: {avg_time_per_interaction*1000:.2f}µs")
        print(f"  Interacciones por segundo: {interactions_per_second:,.0f}")
    
    # Estadísticas de sistema
    if metrics['cpu_usage']:
        print(f"\nUso de recursos del sistema:")
        print(f"  CPU promedio: {np.mean(metrics['cpu_usage']):.1f}%")
        print(f"  CPU máximo: {np.max(metrics['cpu_usage']):.1f}%")
        
    if metrics['memory_usage']:
        print(f"  Memoria promedio: {np.mean(metrics['memory_usage']):.1f}%")
        print(f"  Memoria máximo: {np.max(metrics['memory_usage']):.1f}%")
        
    if metrics['fps']:
        print(f"  FPS promedio: {np.mean(metrics['fps']):.1f}")
        print(f"  FPS mínimo: {np.min(metrics['fps']):.1f}")


def benchmark_parallelization(bodies: List[Body], dt: float = 0.01, 
                            use_rk4: bool = True, steps: int = 50):
    """
    Ejecuta un benchmark comparativo de todos los métodos de paralelización.
    
    Args:
        bodies: Lista de cuerpos para el benchmark
        dt: Paso de tiempo
        use_rk4: Si usar Runge-Kutta 4
        steps: Número de pasos a ejecutar para cada método
    """
    n_bodies = len(bodies)
    interactions = n_bodies * (n_bodies - 1) // 2
    
    print(f"Benchmark con {n_bodies} cuerpos ({interactions:,} interacciones por paso)")
    print(f"Ejecutando {steps} pasos por método...")
    print(f"Método de integración: {'Runge-Kutta 4' if use_rk4 else 'Euler'}")
    print(f"Núcleos disponibles: {mp.cpu_count()}")
    print()
    
    methods = ['serial', 'numba', 'multiprocessing']
    results = {}
    
    for method in methods:
        if method == 'numba' and not NUMBA_AVAILABLE:
            print(f"⚠️  Saltando {method}: Numba no disponible")
            continue
            
        print(f"🚀 Ejecutando método: {method.upper()}")
        
        # Crear copias de los cuerpos para cada test
        test_bodies = []
        for body in bodies:
            test_bodies.append(Body(
                mass=body.mass, x=body.x, y=body.y, 
                vx=body.vx, vy=body.vy, fx=body.fx, fy=body.fy,
                color=body.color, size=body.size
            ))
        
        # Crear simulación
        sim = NBodySimulation(
            bodies=test_bodies,
            dt=dt,
            use_rk4=use_rk4,
            parallel_method=method,
            num_processes=mp.cpu_count()
        )
        
        # Ejecutar benchmark
        start_time = time.time()
        step_times = []
        
        for i in range(steps):
            step_start = time.time()
            sim.calculate_forces()
            if use_rk4:
                sim.integrate_rk4()
            else:
                sim.integrate_euler()
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            if i % 10 == 0:
                print(f"  Paso {i+1}/{steps} - {step_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        avg_step_time = np.mean(step_times)
        
        # Limpiar recursos
        if hasattr(sim, '_process_pool') and sim._process_pool is not None:
            sim._process_pool.shutdown(wait=True)
        
        results[method] = {
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'std_step_time': np.std(step_times),
            'min_step_time': np.min(step_times),
            'max_step_time': np.max(step_times),
            'interactions_per_second': interactions / avg_step_time if avg_step_time > 0 else 0
        }
        
        print(f"  ✅ Completado en {total_time:.2f}s (promedio: {avg_step_time*1000:.2f}ms/paso)")
        print()
    
    # Mostrar resultados comparativos
    print("🏆 RESULTADOS DEL BENCHMARK")
    print("="*80)
    print(f"{'Método':<15} {'Tiempo/Paso':<12} {'Interac/seg':<12} {'Speedup':<8} {'Eficiencia'}")
    print("-"*80)
    
    # Usar serial como referencia para speedup
    baseline_time = results.get('serial', {}).get('avg_step_time', 0)
    
    for method, result in sorted(results.items(), key=lambda x: x[1]['avg_step_time']):
        avg_time = result['avg_step_time']
        speedup = baseline_time / avg_time if avg_time > 0 and baseline_time > 0 else 1.0
        efficiency = speedup / mp.cpu_count() * 100 if method != 'serial' else 100
        
        print(f"{method.upper():<15} "
              f"{avg_time*1000:>8.2f}ms   "
              f"{result['interactions_per_second']:>8.0f}     "
              f"{speedup:>5.2f}x   "
              f"{efficiency:>6.1f}%")
    
    print("-"*80)
    
    # Recomendación
    best_method = min(results.keys(), key=lambda k: results[k]['avg_step_time'])
    best_speedup = baseline_time / results[best_method]['avg_step_time'] if baseline_time > 0 else 1.0
    
    print(f"\n🎯 RECOMENDACIÓN:")
    print(f"   Mejor método: {best_method.upper()}")
    print(f"   Speedup obtenido: {best_speedup:.2f}x")
    print(f"   Ejemplo: python nbody_simulation.py benchmark --bodies {n_bodies} --parallel {best_method}")


def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Simulación N-cuerpos de alta intensidad computacional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Escenarios disponibles:
  solar     - Sistema solar simplificado (25 cuerpos)
    solar     - Sistema solar simplificado (25 cuerpos)
    galaxy    - Cluster galáctico (200+ cuerpos) - MUY intensivo
    collision - Colisión de dos galaxias (visualmente atractivo)
    benchmark - Escenario de benchmark puro (personalizable) - EXTREMADAMENTE intensivo
  benchmark - Escenario de benchmark puro (personalizable) - EXTREMADAMENTE intensivo

Métodos de paralelización:
  numba         - JIT compilation con paralelización automática (RECOMENDADO)
  multiprocessing - Distribución entre procesos (para sistemas con muchos núcleos)
  serial        - Sin paralelización (solo para comparación)

Ejemplos de uso:
  # Básico con paralelización automática
  python nbody_simulation.py solar --rk4
  
  # Cluster galáctico con multiprocessing
  python nbody_simulation.py galaxy --bodies 500 --parallel multiprocessing
  
  # Benchmark extremo con configuración manual
  python nbody_simulation.py benchmark --bodies 1000 --parallel numba --no-trails
  
  # Comparar métodos de paralelización
  python nbody_simulation.py benchmark --bodies 500 --benchmark-parallel
        """)
    
    parser.add_argument('scenario', choices=['solar', 'galaxy', 'collision', 'benchmark'],
                        help='Escenario de simulación a ejecutar')
    parser.add_argument('--bodies', type=int, 
                        help='Número de cuerpos (solo para galaxy y benchmark)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Paso de tiempo para integración (default: 0.01)')
    parser.add_argument('--rk4', action='store_true',
                        help='Usar integración Runge-Kutta 4 (más preciso, más lento)')
    parser.add_argument('--no-trails', action='store_true',
                        help='Desactivar estelas de movimiento')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Desactivar visualización de métricas')
    parser.add_argument('--window', type=float, default=15.0,
                        help='Tamaño de la ventana de visualización (default: 15)')
    parser.add_argument('--save', type=str,
                        help='Guardar animación como video (especificar nombre de archivo)')
    parser.add_argument('--headless', action='store_true',
                        help='Ejecutar sin UI (no muestra ventana)')
    parser.add_argument('--fps', type=int, default=20,
                        help='Cuadros por segundo al guardar video/GIF (default: 20)')
    parser.add_argument('--frames', type=int,
                        help='Número total de frames a renderizar (si se especifica, ignora --duration)')
    parser.add_argument('--duration', type=float,
                        help='Duración en segundos para calcular frames en base a --fps')
    parser.add_argument('--force', action='store_true',
                        help='Omitir confirmaciones para escenarios muy pesados')
    parser.add_argument('--parallel', choices=['numba', 'multiprocessing', 'serial'], 
                        default='numba',
                        help='Método de paralelización (default: numba)')
    parser.add_argument('--processes', type=int, 
                        help='Número de procesos para multiprocessing (default: auto-detectar)')
    parser.add_argument('--benchmark-parallel', action='store_true',
                        help='Comparar todos los métodos de paralelización')
    parser.add_argument('--backend', type=str,
                        help='Backend de Matplotlib (QtAgg, TkAgg, Agg, etc.)')
    parser.add_argument('--viewer', choices=['matplotlib','opencv','web'], default='matplotlib',
                        help='Visor en tiempo real: matplotlib (por defecto), opencv (ventana en vivo), o web (Flask MJPEG)')
    parser.add_argument('--compute-only', action='store_true',
                        help='Ejecutar solo la simulación (sin visualización). Requiere --steps o --duration.')
    parser.add_argument('--steps', type=int,
                        help='Número de pasos de simulación al usar --compute-only')
    # Opciones de visualización
    parser.add_argument('--color-mode', choices=['fixed', 'speed', 'mass'], default='fixed',
                        help='Modo de color: fixed (por cuerpo), speed (por velocidad), mass (por masa)')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap para modos de color numéricos (speed/mass). Default: viridis')
    parser.add_argument('--vectors', action='store_true',
                        help='Dibujar vectores de velocidad (quiver)')
    parser.add_argument('--density', action='store_true',
                        help='Dibujar mapa de densidad (hexbin)')
    parser.add_argument('--blit', action='store_true',
                        help='Habilitar blitting para mejorar FPS (puede no ser compatible con todas las opciones)')
    
    args = parser.parse_args()
    
    # Crear el escenario seleccionado
    print(f"Configurando escenario: {args.scenario}")
    bodies: List[Body] = []
    
    if args.scenario == 'solar':
        bodies = create_solar_system()
    elif args.scenario == 'galaxy':
        n_bodies = args.bodies or 200
        bodies = create_galaxy_cluster(n_bodies)
    elif args.scenario == 'collision':
        # Para este escenario, --bodies controla el total aprox. (dos galaxias)
        total = args.bodies or 800
        per = max(100, total // 2)
        bodies = create_galaxy_collision(n_bodies_per_galaxy=per)
    elif args.scenario == 'benchmark':
        n_bodies = args.bodies or 1000
        if n_bodies > 2000 and not args.force:
            response = input(f"ADVERTENCIA: {n_bodies} cuerpos requerirán "
                           f"{n_bodies*(n_bodies-1)//2:,} cálculos por paso. "
                           "¿Continuar? (y/N): ")
            if response.lower() != 'y':
                sys.exit("Simulación cancelada.")
        bodies = create_benchmark_scenario(n_bodies)
    
    
    # Benchmark de paralelización si se solicita
    if args.benchmark_parallel:
        print("\n🔥 EJECUTANDO BENCHMARK DE PARALELIZACIÓN 🔥")
        print("="*60)
        benchmark_parallelization(bodies, args.dt, args.rk4)
        return
    
    # Crear simulación con paralelización
    simulation = NBodySimulation(
        bodies=bodies,
        dt=args.dt,
        use_rk4=args.rk4,
        max_trail_length=1000,
        parallel_method=args.parallel,
        num_processes=args.processes
    )
    
    # Modo compute-only: sin visualización, enfocado en CPU
    if args.compute_only:
        if not args.steps and not args.duration:
            raise SystemExit("--compute-only requiere --steps o --duration")
        target_steps = args.steps
        if target_steps is None and args.duration is not None:
            # Estimar pasos por segundo a partir de dt (aprox)
            target_steps = max(1, int(args.duration / max(args.dt, 1e-9)))

        print("\nEjecutando en modo compute-only (sin UI)...")
        print("Presiona Ctrl+C para cancelar.\n")
        start = time.time()
        try:
            for i in range(target_steps):
                simulation.step()
                if (i + 1) % max(1, target_steps // 10) == 0:
                    elapsed = time.time() - start
                    print(f"  Progreso: {i+1}/{target_steps} pasos | {elapsed:.2f}s transcurridos")
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.")
        finally:
            print_performance_summary(simulation)
        return

    # Crear visualizador
    visualizer = Visualizer(
        simulation=simulation,
        window_size=args.window,
        show_trails=not args.no_trails,
        show_metrics=not args.no_metrics,
        color_mode=args.color_mode,
        cmap=args.cmap,
        show_vectors=args.vectors,
        show_density=args.density,
        use_blit=args.blit
    )
    
    print(f"\nResumen de configuración:")
    print(f"  Escenario: {args.scenario}")
    print(f"  Número de cuerpos: {len(bodies)}")
    print(f"  Interacciones por paso: {len(bodies)*(len(bodies)-1)//2:,}")
    print(f"  Integración: {'Runge-Kutta 4' if args.rk4 else 'Euler'}")
    print(f"  Paso de tiempo: {args.dt}")
    print(f"  Estelas: {'Sí' if not args.no_trails else 'No'}")
    print(f"  Métricas: {'Sí' if not args.no_metrics else 'No'}")
    
    # Estimar carga computacional
    interactions_per_step = len(bodies) * (len(bodies) - 1) // 2
    if interactions_per_step > 100000:
        print(f"\n⚠️  ADVERTENCIA: Esta configuración requiere {interactions_per_step:,} "
              "cálculos por paso. ¡Será muy intensivo para la CPU!")
    
    print("\nPresiona Ctrl+C para detener la simulación y ver estadísticas.")
    
    try:
        # Iniciar simulación
        visualizer.start(
            interval=50,  # 50ms entre frames = ~20 FPS objetivo (se ajusta si se pasa fps)
            save_animation=bool(args.save),
            filename=args.save or 'nbody_simulation.mp4',
            headless=args.headless,  # permitir ventana + grabación en vivo si --save y no --headless
            fps=args.fps,
            frames=args.frames,
            duration=args.duration,
            viewer=args.viewer
        )
    except KeyboardInterrupt:
        print("\n\nSimulación detenida por el usuario.")
        print_performance_summary(simulation)
    except Exception as e:
        print(f"\nError durante la simulación: {e}")
        print_performance_summary(simulation)


if __name__ == "__main__":
    main()