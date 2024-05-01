import numpy as np
import matplotlib.pyplot as plt

def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi

def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input

    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts

def wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False):
    '''Wiggle plot of a sesimic data section

    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)

    Use the column major order for array as in Fortran to optimal performance.

    The following color abbreviations are supported:

    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========


    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi >= 0,
                         facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

# Definindo as funções

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

# Wavelet Ricker
def Ricker(fs,t):
    R = (1 - 2 * np.pi**2 * fs**2 * t**2 ) * (np.exp(-np.pi**2 * fs**2 * t**2))
    return R

def Ricker2(f_max,t):
    fc = f_max / (3 * np.sqrt(np.pi))
    t0 = 2*np.pi / f_max
    td = t - t0
    R = (1 - (2 * np.pi * ((np.pi * fc * td)**2))) * (np.exp(-np.pi * (np.pi * fc * td)**2))
    
    return R

def plot_wavelet(wavelet, f_max, t, nt, dt):

    freq = np.fft.fftfreq(nt, dt)
    mascara = freq > 0
    Y = np.fft.fft(wavelet)
    Amplitude = np.abs(Y / nt)
    
    # Plot wavelet ricker
    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(10,3))

    ax[0].set_title('Wavelet Ricker', fontsize=12)
    ax[0].plot(t, wavelet, 'b',  label="Ricker \nfs = {} Hz".format(f_max))
    ax[0].set_xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
    ax[0].set_ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
    ax[0].legend(loc='upper right', fontsize=11)

    ax[1].set_title('FFT da Ricker', fontsize=12)
    ax[1].plot(freq[mascara], Amplitude[mascara], 'b',  label="Ricker \nfreq_corte = {} Hz".format(f_max))
    ax[1].set_xlabel('Frequencia (hz)', fontsize=10)
    ax[1].set_ylabel('Amplitude', fontsize=10)  
    ax[1].legend(loc='upper right', fontsize=11)

    plt.show()
    
def reflectivity(velocidade, densidade):
    z = densidade * velocidade
    refl = np.zeros(len(z))

    for i in range(len(z)-1):
        z2 = z[i+1]
        z1 = z[i]
        refl[i] = (z2 - z1) / (z2 + z1)

    return refl

def plot_modelelo_convolucional_1D(VP, RHOB, DEPTH, z, refletividade, trace):
    # Plots
    fig, ax = plt.subplots(ncols=5,nrows=1,figsize=(16,10))

    # Plot Perfil de Velocidade
    ax[0].plot(VP,DEPTH)
    ax[0].set_title('VP')
    ax[0].set_xlabel('Velocidade (m/s)')
    ax[0].set_ylabel('Profundidade (m)')
    ax[0].invert_yaxis()

    # Plot Perfil de Densidade
    ax[1].plot(RHOB,DEPTH)
    ax[1].set_title('RHOB')
    ax[1].set_xlabel('Densidade (kg/m³)')
    ax[1].invert_yaxis()

    ax[2].plot(z,DEPTH)
    ax[2].set_title('Impedância')
    ax[2].set_xlabel('Impedância Acustica')
    ax[2].invert_yaxis()

    # Plot Refletividade de Camadas
    ax[3].plot(refletividade,DEPTH)
    ax[3].set_title('Refletividade')
    ax[3].set_xlabel('Refletividade (kg/m³)')
    ax[3].invert_yaxis()

    #ax[4].plot(trace, DEPTH,'b', label='Ricker {} Hz'.format(fs))
    ax[4].plot(trace, DEPTH[0:], lw=1, color='black')  
    ax[4].fill_betweenx(DEPTH[0:], trace, 0., trace > 0, color='black')
    #ax.fill_betweenx(depth1[0:], trace, 0., trace < 0, color='red')
    ax[4].set_title('Traço Sismico')
    ax[4].set_xlabel('Traço Sismico')
    ax[4].invert_yaxis()


    plt.tight_layout()
    plt.show()
    
def ajustes(depth_min, depth_max):
    dado1 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_1.las.txt', skiprows=37)
    dado2 = np.loadtxt('seismic_modeling/modelagem_convolucional1D/dados/1ess53ess_2.las.txt', skiprows=37)
    
    DT_min = np.array(np.where(dado1[:, 0] >= depth_min))[0][0]
    DT_max = np.array(np.where(dado1[:, 0] <= depth_max))[0][-1]

    RHOB_min = np.array(np.where(dado2[:, 0] >= depth_min))[0][0]
    RHOB_max = np.array(np.where(dado2[:, 0] <= depth_max))[0][-1]

    D_min = np.array(np.where(dado1[:, 0] >= depth_min))[0][0]
    D_max = np.array(np.where(dado1[:, 0] <= depth_max))[0][-1]

    DEPTH1 = dado1[D_min:D_max + 1,0]
    DT = dado1[DT_min:DT_max + 1,1]

    DEPTH2  = dado2[(dado2[:, 0] >= depth_min) & (dado2[:, 0] <= depth_max)]
    RHOB = dado2[RHOB_min:RHOB_max + 1,1] * 1000

    VP = (304.8 / DT) * 1000
    
    return VP, RHOB, DEPTH1 

def plot_fft(data, dt):
    # Trasnformada de Fourier da Ricker
    freq_trace = np.fft.fftfreq(len(data), dt)
    mascara_trace = freq_trace > 0
    Y_trace = np.fft.fft(data)
    Amplitude_trace = np.abs(Y_trace / len(data))

    fig, ax = plt.subplots()

    ax.set_title("Espectro de Frequência", fontsize=12)
    ax.plot(freq_trace[mascara_trace], Amplitude_trace[mascara_trace], 'b')
    ax.set_xlabel('Frequencia (hz)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    #ax.set_xlim(0,50)  
    ax.legend(loc='upper right', fontsize=11)
    plt.show()
    