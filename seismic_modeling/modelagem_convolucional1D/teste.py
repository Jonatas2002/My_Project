import numpy as np

# Carregar os dados dos arquivos
dado1 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_1.las.txt', skiprows=37)
dado2 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_2.las.txt', skiprows=37)

# Definir o intervalo de profundidade desejado
depth_min = 2836
depth_max = 4334

prof1 = dado1[(dado1[:, 0] >= depth_min) & (dado1[:, 0] <= depth_max)]
prof2 = dado2[(dado2[:, 0] >= depth_min) & (dado2[:, 0] <= depth_max)]


# Calcular os índices mínimos e máximos para os dados DT e RHOB
DT_min_index = np.where(dado1[:, 0] >= depth_min)[0][0]
DT_max_index = np.where(dado1[:, 0] <= depth_max)[0][-1]
RHOB_min_index = np.where(dado2[:, 0] >= depth_min)[0][0]
RHOB_max_index = np.where(dado2[:, 0] <= depth_max)[0][-1]

# Filtrar os dados de DT e RHOB usando os índices calculados
DT = dado1[DT_min_index:DT_max_index + 1, 1]  # Adicionamos 1 ao DT_max_index para incluir o último elemento
RHOB = dado2[RHOB_min_index:RHOB_max_index + 1, 1]  # Adicionamos 1 ao RHOB_max_index para incluir o último elemento

# Saída dos índices mínimos e máximos para verificação



print('dt', len(DT))
print('RHOB', len(RHOB))
print('depth1', len(prof1))
print('depth2', len(prof2))
