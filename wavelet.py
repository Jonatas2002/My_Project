# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# wavelet ormsby
def  Ormsby(f1, f2, f3, f4, t):
    w1 = (((np.pi * f4)**2)/(np.pi*f4 - np.pi*f3)) * np.sinc(f4 * t)**2
    w2 = (((np.pi * f3)**2)/(np.pi*f4 - np.pi*f3)) * np.sinc(f3 * t)**2
    w3 = (((np.pi * f2)**2)/(np.pi*f2 - np.pi*f1)) * np.sinc(f2 * t)**2
    w4 = (((np.pi * f1)**2)/(np.pi*f2 - np.pi*f1)) * np.sinc(f1 * t)**2

    O = ((w1 - w2) - (w3 - w4))
    return O

# Parametros geral
T = 1   # tempo em segundos
dt = 0.004  # taxa de amostragem
#n = int((T/dt)+1) # numero de amostra
#t = np.linspace(0, T, n, endpoint=False)   #base de tempo
#tlag= 0.5 # Deslocamento no tempo em segundo


#Parametros de frequencia Ormsby
f1 = 5   # frequência de corte baixo
f2 = 10  # frequência passa-baixa
f3 = 40  # frequência passa-alta
f4 = 45  # frequência de corte alta

nt = 101
t = (np.arange(nt) * dt) -0.2
#t = np.linspace(-0.2,0.2,50)
#t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

#Função Wavelet Ormsby
O = Ormsby(f1, f2, f3, f4, t)

# Trasnformada de Fourier da Ricker
freq = np.fft.fftfreq(nt, dt)
mascara = freq > 0
Y = np.fft.fft(O)
Amplitude = np.abs(Y / nt)


plt.figure(figsize=(18, 6))
plt.suptitle("visualização gráfica da wavelet ormsby e sua integral a esquerda, acompanhadas dos espectros de frequência a direita", fontsize=16)

# plot wavalet ormsby
plt.subplot(1,2,1)
plt.title('Função Wavelet Ormsby', fontsize=12)
plt.plot(t, O, 'b-', label= 'Ormsby \nf1 = {} Hz \nf2 = {} Hz \nf3 = {} Hz \nf4 = {} Hz'.format(f1, f2, f3, f4))
plt.grid()
plt.xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
plt.ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
plt.legend(loc='upper right', fontsize=11)

plt.subplot(1,2,2)
plt.title('Espectro de Frequencia - Wavelet Ormsby', fontsize=12)
plt.plot(freq[mascara], Amplitude[mascara], 'b-', label= 'Ormsby \nf1 = {} Hz \nf2 = {} Hz \nf3 = {} Hz \nf4 = {} Hz'.format(f1, f2, f3, f4))
plt.grid()
plt.xlabel('Frequência (Hz)', fontsize=10)  # legenda do eixo x
plt.ylabel('|X(f)|', fontsize=10)  # legenda do eixo y
plt.legend(loc='upper right', fontsize=11)



##############################################################################
plt.tight_layout()
plt.show()
