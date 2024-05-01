import numpy as np
import matplotlib.pyplot as plt

from function import Ricker
from function import Ricker2
from function import plot_wavelet
from function import reflectivity
from function import plot_modelelo_convolucional_1D
from function import wiggle
from function import ajustes
from function import plot_fft
# Parametros da modelagem
TIME = 1                                        # tempo em segundos
dt = 0.002                                      # taxa de amostragem
fs = 25                                         # Frequencia do sinal ricker

depth_min = 2836                                # PROF_MIN = 2836
depth_max = 4334                                # PROF_MAX = 4334

nt = int((TIME/dt) + 1)                         # numero de amostra
t = np.linspace(0, TIME, nt, endpoint=False)    # base de tempo

# Função Wavelet Ricker
R = Ricker2(fs,t)
plot_wavelet(R, fs, t, nt, dt)

# abrindo os arquivos
vel, rhob, depth1 = ajustes(depth_min, depth_max)

# Calculo da Impedância,  Refletividade e traço sismico
z = vel*rhob 
refletividade =  reflectivity(vel, rhob)
trace = np.convolve(R, refletividade, mode='same')

plot_modelelo_convolucional_1D(vel, rhob, depth1, z, refletividade, trace)

plot_fft(trace, dt)

plot_fft(refletividade, dt)


plt.figure()
plt.hist(refletividade,bins=1000)
plt.show()

#----------------------------------------------------------------------------------------------------------------
"""Modelagem Convolucional 2D - Wiggle"""
nx = 20  # quantidade de traços sismicos
trace2D = (np.array([trace]*nx).T)

plt.figure(figsize=(10, 10))
plt.title("Plot Wiggle")
wiggle(trace2D, depth1, xx=None, color='k', sf=0.15, verbose=False)
plt.xlabel("Traços")
plt.ylabel("Profundidade")
plt.tight_layout()
plt.show()

