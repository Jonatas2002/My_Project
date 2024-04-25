# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Wavelet Ricker
def Ricker(fs,t):
    R = (1 - 2 * np.pi**2 * fs**2 * t**2 ) * (np.exp(-np.pi**2 * fs**2 * t**2))
    return R

# Parametros geral
T = 1   # tempo em segundos
dt = 0.002  # taxa de amostragem
nt = int((T/dt) + 1) # numero de amostra
t = np.linspace(0, T, nt, endpoint=True)   #base de tempo
t_lag = 0.5
fs = 10  #frequencia do sinal ricker

# Função Wavelet Ricker
wavelet = Ricker(fs, t-t_lag)

# PLOT DO GRAFICOS
plt.figure(figsize=(5,3))
plt.title('Função Wavelet Ricker', fontsize=12)
plt.plot(t, wavelet, 'b',  label="Ricker \nfs = {} Hz".format(fs))
plt.xlabel('tempo (s)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)  
plt.legend(loc='upper right', fontsize=11)

plt.show()
