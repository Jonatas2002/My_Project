import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definindo os parâmetros
nt = 101  # Número de pontos em t
freq = 10  # Frequência da onda em Hz
lambida = 1
w = 2 * np.pi * freq
k = 2 * np.pi / lambida # Número de onda
duration = 5  # Duração da animação em segundos
fps = 30  # Quadros por segundo

t = np.linspace(0, 1, nt)

# Função para atualizar o plot a cada quadro da animação
def update(frame):
    plt.cla()  # Limpa o plot anterior
    y = np.sin(k * t - w * frame / (duration / 2 * np.pi))  # Calcula a onda senoidal para o tempo atual
    plt.plot(t, y)
    plt.xlabel('Posição (x)')
    plt.ylabel('Amplitude')
    plt.title('Onda Senoidal em 1D')
    #plt.ylim(-1.5, 1.5)  # Define os limites do eixo y
    plt.grid(True)  # Adiciona uma grade ao plot

# Criação da animação
fig = plt.figure()
ani = FuncAnimation(fig, update, frames=np.linspace(0, duration, int(duration * fps)), interval=1000 / fps)

plt.show()
