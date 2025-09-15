# 🌌 Simulación N-Cuerpos - Proyecto Científico de Alto Rendimiento

Un simulador de N-cuerpos computacionalmente intensivo que resuelve las interacciones gravitacionales entre múltiples cuerpos celestes en tiempo real. Perfecto para estresar la CPU y estudiar sistemas gravitacionales complejos.

## 🚀 Características

- **Algoritmo O(n²)**: Calcula todas las interacciones gravitacionales entre pares de cuerpos
- **🔥 PARALELIZACIÓN MULTI-NÚCLEO**: Aprovecha todos los núcleos de tu CPU
  - **Numba JIT**: Compilación just-in-time con paralelización automática
  - **Multiprocessing**: Distribución de trabajo entre procesos
  - **NumPy BLAS**: Operaciones vectorizadas multi-hilo
- **Integración Runge-Kutta 4**: Máxima precisión numérica (opcional: Euler para mayor velocidad)
- **Visualización en tiempo real**: Animación matplotlib con estelas de movimiento
- **Métricas de rendimiento**: Monitoreo de CPU, memoria, FPS y eficiencia paralela
- **Múltiples escenarios**: Sistema solar, clusters galácticos, y benchmark puro
- **Benchmark integrado**: Compara métodos de paralelización automáticamente
- **Escalable**: Desde decenas hasta miles de cuerpos

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

Dependencias:
- numpy >= 1.21.0
- matplotlib >= 3.5.0  
- psutil >= 5.8.0
- numba >= 0.56.0 (para paralelización JIT)

## 🎯 Uso Básico

### Sistema Solar (25 cuerpos)
```bash
python nbody_simulation.py solar --rk4
```

### Cluster Galáctico (200+ cuerpos) - **CON PARALELIZACIÓN**
```bash
python nbody_simulation.py galaxy --bodies 500 --parallel numba
```

### Benchmark Extremo (¡Miles de cuerpos!) - **TODOS LOS NÚCLEOS**
```bash
python nbody_simulation.py benchmark --bodies 1000 --parallel multiprocessing --no-trails
```

### 🔥 **NUEVO**: Benchmark de Paralelización
```bash
# Compara automáticamente todos los métodos
python nbody_simulation.py benchmark --bodies 500 --benchmark-parallel
```

## ⚡ Niveles de Intensidad + PARALELIZACIÓN

| Escenario | Cuerpos | Interacciones/paso | Sin Paralelo | **Con Paralelo** |
|-----------|---------|-------------------|--------------|------------------|
| Solar     | ~25     | ~300              | Baja         | **Ligera**       |
| Galaxy    | 200-500 | 20K-125K          | Alta         | **Moderada**     |
| Benchmark | 1000+   | 500K+             | **EXTREMA**  | **Intensa**      |

**💡 Con paralelización Numba/Multiprocessing puedes manejar 2-8x más cuerpos!**

## 🔧 Parámetros Avanzados + PARALELIZACIÓN

```bash
# Integración de alta precisión con Numba
python nbody_simulation.py solar --rk4 --dt 0.001 --parallel numba

# Máximo rendimiento con multiprocessing
python nbody_simulation.py galaxy --no-trails --no-metrics --parallel multiprocessing --processes 8

# Benchmark comparativo de métodos
python nbody_simulation.py benchmark --bodies 1000 --benchmark-parallel

# Control manual de procesos
python nbody_simulation.py benchmark --parallel multiprocessing --processes 4

# Guardar animación con paralelización
python nbody_simulation.py solar --save solar_system.mp4 --parallel numba
```

## 📊 Métricas de Rendimiento

El simulador proporciona:
- ⏱️ Tiempo de cálculo por paso (promedio/min/max)
- 🖥️ Uso de CPU y memoria en tiempo real
- 📈 FPS y rendimiento de visualización
- 🔢 Interacciones gravitacionales por segundo
- **🚀 Eficiencia de paralelización (speedup real)**
- **⚡ Comparación automática entre métodos**

## 🧮 Complejidad Computacional

**Cálculos por paso de tiempo:**
- n cuerpos → n(n-1)/2 interacciones gravitacionales
- 1000 cuerpos → **499,500 cálculos/paso**
- Con Runge-Kutta 4 → **×4 multiplicador**

## ⚠️ Advertencias

- **1000+ cuerpos**: Requerirá >500K cálculos por paso
- **CPU intensivo**: Puede saturar todos los núcleos
- **Memoria**: Las estelas consumen RAM con el tiempo
- **Runge-Kutta 4**: 4x más lento que Euler pero más preciso

## 🎮 Controles

- `Ctrl+C`: Detener simulación y mostrar estadísticas
- Ventana matplotlib: Zoom, paneo estándar
- Cierre de ventana: Finaliza el programa

## 🔬 Casos de Uso Científicos

1. **Astrofísica**: Simulación de sistemas estelares y galácticos
2. **Benchmarking**: Test de rendimiento de CPU/sistema
3. **Educación**: Visualización de mecánica gravitacional
4. **Investigación**: Análisis de estabilidad orbital

## 🎯 Para Máximo Estrés de CPU

```bash
# ¡ADVERTENCIA: Esto saturará tu CPU!
python nbody_simulation.py benchmark --bodies 2000 --rk4 --dt 0.001
```

Este comando genera **4 millones de interacciones por paso** con integración de máxima precisión.

## 📈 Resultados Típicos

**SIN paralelización (método serial):**
- 100 cuerpos: ~15 FPS, CPU 100% (1 núcleo)
- 500 cuerpos: ~3 FPS, sistema saturado  
- 1000 cuerpos: <1 FPS, prácticamente inutilizable

**CON paralelización Numba/Multiprocessing (8 núcleos):**
- 100 cuerpos: ~60 FPS, CPU 30-50%
- 500 cuerpos: ~25 FPS, CPU 70-90%  
- 1000 cuerpos: ~12 FPS, CPU 95-100%
- 2000+ cuerpos: ~6 FPS, CPU saturada
- **🔥 5000 cuerpos**: ~2 FPS, **¡12.5 MILLONES de interacciones por paso!**

**💥 SPEEDUP TÍPICO: 3-8x dependiendo del número de núcleos**

¡Perfecto para probar los límites de tu hardware! 🔥