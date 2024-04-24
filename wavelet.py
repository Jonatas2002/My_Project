import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definindo a função Ormsby
def Ormsby(f1, f2, f3, f4, t):
    w1 = (((np.pi * f4)**2)/(np.pi*f4 - np.pi*f3)) * np.sinc(f4 * t)**2
    w2 = (((np.pi * f3)**2)/(np.pi*f4 - np.pi*f3)) * np.sinc(f3 * t)**2
    w3 = (((np.pi * f2)**2)/(np.pi*f2 - np.pi*f1)) * np.sinc(f2 * t)**2
    w4 = (((np.pi * f1)**2)/(np.pi*f2 - np.pi*f1)) * np.sinc(f1 * t)**2

    O = ((w1 - w2) - (w3 - w4))
    return O

# Parâmetros da onda Ormsby
dt = 0.004
f1 = 5   # frequência de corte baixo
f2 = 10  # frequência passa-baixa
f3 = 40  # frequência passa-alta
f4 = 45  # frequência de corte alta
nt = 101
t = (np.arange(nt) * dt)
t_lag = 0.2
# Criação da figura e dos subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Inicialização da linha da onda Ormsby
line, = axs[0].plot([], [], 'b-', lw=2)
axs[0].set_xlim(0, 0.4)
axs[0].set_ylim(-100, 220)
axs[0].set_title('Wavelet Ormsby')
axs[0].set_xlabel('Tempo (s)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Inicialização da linha do espectro de frequência
line_spectrum, = axs[1].plot([], [], 'b-', lw=2)
axs[1].set_xlim(0, 120)
axs[1].set_ylim(0, 9)
axs[1].set_title('Espectro de Frequência - Wavelet Ormsby')
axs[1].set_xlabel('Frequência (Hz)')
axs[1].set_ylabel('|X(f)|')
axs[1].grid(True)

# Função de inicialização da animação
def init():
    line.set_data([], [])
    line_spectrum.set_data([], [])
    return line, line_spectrum

# Função de atualização da animação
def update(frame):
    t_current = (t / nt) * frame  # Calcula o tempo atual
    y = Ormsby(f1, f2, f3, f4, t_current-t_lag)  # Calcula a onda Ormsby para o tempo atual
    freq = np.fft.fftfreq(nt, dt)
    Y = np.fft.fft(y)
    Amplitude = np.abs(Y / nt)
    
    line.set_data(t_current, y)
    line_spectrum.set_data(freq[freq > 0], Amplitude[freq > 0])
    return line, line_spectrum

# Criação da animação
ani = FuncAnimation(fig, update, frames=nt, init_func=init, blit=True)

plt.tight_layout()
plt.show()
