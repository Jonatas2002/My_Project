'''FFT inversa e Convolução
Objetivo: Realizar a transformada inversa de Fourier de sinal ja conhecido
e comparar a convolução de dois sinais no dominio do tempo e da frequência

Status do Projeto:
Transformada de Fourier: OK
Transformada Inversa de Fourier - OK
Convolução no dominio do Tempo - OK
Convolução no dominio da Frequencia - OK

Autor: Jonatas Oliveira de Araujo '''

import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
T = 1                       # Tempo (s)
dt = 1/128                  # Taxa de Amostragem
fs1 = 10                    # Frequência do sinal 1
fs2 = 40                    # Frequência do sinal 2
nt = int((T/dt) + 1)        # Numeros de Amostra
time = np.arange(nt) * dt   # Base de Tempo

signal1 = np.sin(time * 2*np.pi * fs1)
signal2 = np.cos(time * 2*np.pi * fs2)

# Convolução no dominio do tempo
signal3 = np.convolve(signal1,signal2, 'same')

# Trasnformada de Fourier
freq = np.fft.fftfreq(nt, dt)
mascara = freq > 0

# Signal 1
Y1 = np.fft.fft(signal1)
Amp1 = np.abs(Y1)

# Signal 2
Y2 = np.fft.fft(signal2)
Amp2 = np.abs(Y2)

# Signal 1
Y3 = np.fft.fft(signal3)
Amp3 = np.abs(Y3)


""" Convolução no dominio da Frequencia"""
S_con = Y1 * Y2

signal4 = np.real(np.fft.ifft(S_con))
Y4 = np.fft.fft(signal4)
Amp4 = np.abs(Y4)


# Plot dos Graficos
fig, ax = plt.subplots(ncols=2, nrows=2, num= "BASE SIGNAL", figsize=(14, 6))
ax[0, 0].set_title('Signal 1 - Time Domain', fontsize=15)
ax[0, 0].plot(time, signal1)
ax[0, 0].set_xlabel("Time [s]", fontsize=10)
ax[0, 0].set_ylabel(r"$E_x(t)$", fontsize=10)

ax[0, 1].set_title("Signal 1 -Fast Fourier Transform", fontsize=15)
ax[0, 1].plot(freq[mascara], Amp1[mascara])
ax[0, 1].set_xlabel("Frequency [Hz]", fontsize=10)
ax[0, 1].set_ylabel(r"$|E_x(f)|$", fontsize=10)

ax[1, 0].set_title('Signal 2 - Time Domain', fontsize=15)
ax[1, 0].plot(time, signal2)
ax[1, 0].set_xlabel("Time [s]", fontsize=10)
ax[1, 0].set_ylabel(r"$E_x(t)$", fontsize=10)

ax[1, 1].set_title("Signal 2 -Fast Fourier Transform", fontsize=15)
ax[1, 1].plot(freq[mascara], Amp2[mascara])
ax[1, 1].set_xlabel("Frequency [Hz]", fontsize=10)
ax[1, 1].set_ylabel(r"$|E_x(f)|$", fontsize=10)

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(ncols=2, nrows=2, num= "CONVOLUTION IN THE TIME DOMAIN vs CONVOLUTION IN THE FREQUENCY DOMAIN", figsize=(14, 6))
ax[0, 0].set_title('Signal 3 - CONVOLUTION IN THE TIME DOMAIN', fontsize=15)
ax[0, 0].plot(time, signal3)
ax[0, 0].set_xlabel("Time [s]", fontsize=10)
ax[0, 0].set_ylabel(r"$E_x(t)$", fontsize=10)

ax[0, 1].set_title("Signal 3 - CONVOLUTION IN THE TIME DOMAIN", fontsize=15)
ax[0, 1].plot(freq[mascara], Amp3[mascara])
ax[0, 1].set_xlabel("Frequency [Hz]", fontsize=10)
ax[0, 1].set_ylabel(r"$|E_x(f)|$", fontsize=10)

ax[1, 0].set_title('Signal 4 - CONVOLUTION IN THE FREQUENCY DOMAIN', fontsize=15)
ax[1, 0].plot(time, signal4)
ax[1, 0].set_xlabel("Time [s]", fontsize=10)
ax[1, 0].set_ylabel(r"$E_x(t)$", fontsize=10)

ax[1, 1].set_title("Signal 4 - CONVOLUTION IN THE FREQUENCY DOMAIN", fontsize=15)
ax[1, 1].plot(freq[mascara], Amp4[mascara])
ax[1, 1].set_xlabel("Frequency [Hz]", fontsize=10)
ax[1, 1].set_ylabel(r"$|E_x(f)|$", fontsize=10)

fig.tight_layout()
plt.show()