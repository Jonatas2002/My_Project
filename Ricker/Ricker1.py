# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

def Ricker(f_max,t):
    fc = f_max / (3 * np.sqrt(np.pi))
    t0 = 2*np.pi / f_max
    td = t - t0
    R = (1 - (2 * np.pi * ((np.pi * fc * td)**2))) * (np.exp(-np.pi * (np.pi * fc * td)**2))
    
    return R


# Parametros geral
T = 1   # tempo em segundos
dt = 0.002  # taxa de amostragem
nt = int((T/dt)+1) # numero de amostra
t = np.linspace(0, T, nt, endpoint=True)   #base de tempo

f_corte = 25  #frequencia do sinal ricker

# Função Wavelet Ricker
R = Ricker(f_corte, t)

# Trasnformada de Fourier da Ricker
freq = np.fft.fftfreq(nt, dt)
mascara = freq > 0
Y = np.fft.fft(R)
Amplitude = np.abs(Y / nt)

# PLOT DOS GRAFICOS
plt.figure(figsize=(10,3))

# Plot wavelet ricker
plt.subplot(121)
plt.title('Função Wavelet Ricker', fontsize=12)
plt.plot(t, R, 'b',  label="Ricker \nfs = {} Hz".format(f_corte))
plt.grid()
plt.xlabel('tempo (s)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)  
plt.legend(loc='upper right', fontsize=11)

plt.subplot(122)
plt.title('FFT da Ricker', fontsize=12)
plt.plot(freq[mascara], Amplitude[mascara], 'b',  label="Ricker \nfs = {} Hz".format(f_corte))
plt.grid()
plt.xlabel('Frequencia (hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)  
plt.legend(loc='upper right', fontsize=11)

plt.show()

print(nt)