import numpy as np

# Wavelet Ricker
def Ricker(fs,t):
    R = (1 - 2 * np.pi**2 * fs**2 * t**2 ) * (np.exp(-np.pi**2 * fs**2 * t**2))
    return R


# Calculo da Refletividade
def reflectivity(velocidade, densidade):
    z = densidade * velocidade
    refl = np.zeros(len(z))

    for i in range(len(z)-1):
        z2 = z[i+1]
        z1 = z[i]
        refl[i] = (z2 - z1) / (z2 + z1)

    return refl