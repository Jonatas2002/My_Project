import numpy as np
import matplotlib.pyplot as plt
from function import Ricker
from function import Ricker2
from function import reflectivity
from function import wiggle

# Parametros da wavelet
T = 1   # tempo em segundos
dt = 1/500  # taxa de amostragem
nt = int((T/dt) + 1) # numero de amostra
t = np.linspace(0, T, nt, endpoint=False)   #base de tempo
#tlag= 0.5 # Deslocamento no tempo em segundo

fs = 25  #frequencia do sinal ricker

# Função Wavelet Ricker
R = Ricker2(fs, t)

# Trasnformada de Fourier da Ricker
freq = np.fft.fftfreq(nt, dt)
mascara = freq > 0
Y = np.fft.fft(R)
Amplitude = np.abs(Y / nt)

# Plot wavelet ricker
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title('Wavelet Ricker', fontsize=12)
plt.plot(t, R, 'b',  label="Ricker \nfs = {} Hz".format(fs))
plt.xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
plt.ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
plt.legend(loc='upper right', fontsize=11)

plt.subplot(122)
plt.title('FFT da Ricker', fontsize=12)
plt.plot(freq[mascara], Amplitude[mascara], 'b',  label="Ricker \nfreq_corte = {} Hz".format(fs))
plt.xlabel('Frequencia (hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)  
plt.legend(loc='upper right', fontsize=11)

plt.show()

# abrindo os arquivos
dado1 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_1.las.txt', skiprows=37)
dado2 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_2.las.txt', skiprows=37)

# convertendo para o sistema SI
depth1 = dado1[11741:,0]
dt = dado1[11741:,1]

vel = (304.8 / dt) * 1000

depth2 = dado2[:-45,0]
rhob = dado2[:-45,1] * 1000

# Calculo da Impedância e Refletividade
z = vel*rhob  # Calculo da Impedância

# Calculo da Refletividade
refletividade =  reflectivity(vel, rhob)

# Convolvendo a refletividade com cada uma das wavelets e gerendo traços sismicos sinteticos
trace = np.convolve(R, refletividade, mode='same')   # Convolução da Refletividade com a wavelet ricker

# Plots
plt.figure(figsize=(18, 10))

# Plot Perfil de Velocidade
plt.subplot(1,5,1)
plt.plot(vel,depth1)
plt.title('Velocidade')
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Tempo (s)')
plt.gca().invert_yaxis()

# Plot Perfil de Densidade
plt.subplot(1,5,2)
plt.plot(rhob,depth1)
plt.title('Densidade')
plt.xlabel('Densidade (kg/m³)')
plt.ylabel('Tempo (s)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.gca().invert_yaxis()

plt.subplot(1,5,3)
plt.plot(z,depth1)
plt.title('Impedância')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Impedância Acustica')
#plt.ylabel('Tempo (s)')
plt.gca().invert_yaxis()

# Plot Refletividade de Camadas
plt.subplot(1,5,4)
plt.plot(refletividade,depth1)
plt.title('Refletividade')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade (kg/m³)')
#plt.ylabel('Tempo (s)')
plt.gca().invert_yaxis()

plt.subplot(1,5,5)
plt.plot(trace,depth1,'b', label='Ricker {} Hz'.format(fs))
plt.title('Traço Sismico (Ricker)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Traço Sismico')
#plt.ylabel('Tempo (s)')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()

# plt.imshow(np.array([trace]*n).T, aspect='auto',
#            extent=(np.min(trace),np.max(trace),
#            np.max(depth1), np.min(depth1)), cmap='Greys')

plt.tight_layout()
plt.show()

# Trasnformada de Fourier da Ricker
freq_trace = np.fft.fftfreq(len(trace), 1/500)
mascara_trace = freq_trace > 0
Y_trace = np.fft.fft(trace)
Amplitude_trace = np.abs(Y_trace / len(trace))

plt.figure()
plt.title('FFT do Traço', fontsize=12)
plt.plot(freq_trace[mascara_trace], Amplitude_trace[mascara_trace], 'b',  label="Ricker \nfs = {} Hz".format(fs))
plt.xlabel('Frequencia (hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)  
plt.legend(loc='upper right', fontsize=11)
plt.show()


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
"""Modelagem Convolucional 2D - Wiggle"""
nx = 50  # quantidade de traços sismicos
trace2D = (np.array([trace]*nx).T)

plt.figure(figsize=(15, 10))
plt.title("Plot Wiggle")
wiggle(trace2D, depth1, xx=None, color='k', sf=0.15, verbose=False)
plt.xlabel("Traços")
plt.ylabel("Profundidade")
plt.tight_layout()
plt.show()

