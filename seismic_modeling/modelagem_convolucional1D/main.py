import numpy as np
import matplotlib.pyplot as plt
from function import Ricker
from function import reflectivity

# Parametros da wavelet
T = 1   # tempo em segundos
dt = 0.002  # taxa de amostragem
n = int((T/dt) + 1) # numero de amostra
t = np.linspace(0, T, n, endpoint=False)   #base de tempo
tlag= 0.5 # Deslocamento no tempo em segundo

fs = 30  #frequencia do sinal ricker

# Função Wavelet Ricker
R = Ricker(fs, t-tlag)

# Plot wavelet ricker
plt.figure()
plt.title('Função Wavelet Ricker', fontsize=12)
plt.plot(t, R, 'b',  label="Ricker \nfs = {} Hz".format(fs))

plt.xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
plt.ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
plt.legend(loc='upper right', fontsize=11)
plt.show()

# abrindo os arquivos
dado1 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_1.las.txt', skiprows=37)
dado2 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_2.las.txt', skiprows=37)

# convertendo para o sistema SI
depth1 = dado1[11741:,0]
dt = dado1[11741:,1]
vel = dt * 3.2808*10**(-6)

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
plt.title('Perfil Velocidade de Camadas')
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Tempo (s)')
plt.gca().invert_yaxis()

# Plot Perfil de Densidade
plt.subplot(1,5,2)
plt.plot(rhob,depth1)
plt.title('Perfil Densidade de Camadas')
plt.xlabel('Densidade (kg/m³)')
plt.ylabel('Tempo (s)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.gca().invert_yaxis()

plt.subplot(1,5,3)
plt.plot(z,depth1)
plt.title('Impedância Acustica')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Impedância Acustica')
#plt.ylabel('Tempo (s)')
plt.gca().invert_yaxis()

# Plot Refletividade de Camadas
plt.subplot(1,5,4)
plt.plot(refletividade,depth1)
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade (kg/m³)')
#plt.ylabel('Tempo (s)')
plt.gca().invert_yaxis()

plt.subplot(1,5,5)
plt.plot(trace,depth1,'b', label='Ricker {} Hz'.format(fs))
plt.title('Traço Sismico 1 (Ricker)')
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