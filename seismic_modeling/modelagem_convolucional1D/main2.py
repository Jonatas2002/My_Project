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
fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(10,3))

ax[0].set_title('Wavelet Ricker', fontsize=12)
ax[0].plot(t, R, 'b',  label="Ricker \nfs = {} Hz".format(fs))
ax[0].set_xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
ax[0].set_ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
ax[0].legend(loc='upper right', fontsize=11)

ax[1].set_title('FFT da Ricker', fontsize=12)
ax[1].plot(freq[mascara], Amplitude[mascara], 'b',  label="Ricker \nfreq_corte = {} Hz".format(fs))
ax[1].set_xlabel('Frequencia (hz)', fontsize=10)
ax[1].set_ylabel('Amplitude', fontsize=10)  
ax[1].legend(loc='upper right', fontsize=11)

plt.show()

# abrindo os arquivos
dado1 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_1.las.txt', skiprows=37)
dado2 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_2.las.txt', skiprows=37)

# depth_min = 2836
# depth_max = 4334
depth_min = 2836
depth_max = 3000


DT_min = np.array(np.where(dado1[:, 0] >= depth_min))[0][0]
DT_max = np.array(np.where(dado1[:, 0] <= depth_max))[0][-1]

RHOB_min = np.array(np.where(dado2[:, 0] >= depth_min))[0][0]
RHOB_max = np.array(np.where(dado2[:, 0] <= depth_max))[0][-1]

#depth1 = dado1[(dado1[:, 0] >= depth_min) & (dado1[:, 0] <= depth_max)]
D_min = np.array(np.where(dado1[:, 0] >= depth_min))[0][0]
D_max = np.array(np.where(dado1[:, 0] <= depth_max))[0][-1]

depth1 = dado1[D_min:D_max + 1,0]
DT = dado1[DT_min:DT_max + 1,1]

depth2 = dado2[(dado2[:, 0] >= depth_min) & (dado2[:, 0] <= depth_max)]
rhob = dado2[RHOB_min:RHOB_max + 1,1] * 1000
vel = (304.8 / DT) * 1000

print(np.min(depth1))

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
#plt.ylim(3400,3200)
plt.gca().invert_yaxis()

plt.subplot(1,5,5)
plt.plot(trace,depth1,'b', label='Ricker {} Hz'.format(fs))
plt.title('Traço Sismico (Ricker)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Traço Sismico')
#plt.ylabel('Tempo (s)')
#plt.ylim(3400,3200)

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
nx = 2  # quantidade de traços sismicos
trace2D = (np.array([trace]*nx).T)

plt.figure(figsize=(3, 10))
plt.title("Plot Wiggle")
wiggle(trace2D, depth1, xx=None, color='k', sf=0.15, verbose=False)
plt.xlabel("Traços")
plt.ylabel("Profundidade")
#plt.ylim(3400,3200)
plt.tight_layout()
#plt.show()

fig, ax = plt.subplots(figsize=(2,9), sharey=True)
      
ax.set_xlabel("Synthetic seismogram")
ax.plot(trace, depth1[0:], lw=1, color='black')  
ax.fill_betweenx(depth1[0:], trace, 0., trace > 0, color='black')
#ax.fill_betweenx(depth1[0:], trace, 0., trace < 0, color='red')


#ax.set_ylim(3400,3200)
plt.show()

