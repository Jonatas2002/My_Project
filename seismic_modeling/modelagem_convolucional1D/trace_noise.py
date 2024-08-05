import numpy as np
import matplotlib.pyplot as plt
import function as func


# # ----------------------------------------------------------------------
# # ----------------------------PARAMETROS -------------------------------
nt = 501
dx = 1
time = 0.8
dt = time/nt
t = np.arange(nt) * dt

fs = 25
tlag = 0.4


# ----------------------------------------------------------------------
# -------------------MODELAGEM CONVOLUCIONAL 1D ------------------------

wavelet = func.Ricker(fs, t-tlag)

interface = np.array([0.215, 0.425, 1])
vp = np.array([1500, 1000, 3000])
rhob = np.array([1000, 1500, 2600])

vp_model = func.model_1D(nt, dt, interface, vp)
rhob_model = func.model_1D(nt, dt, interface, rhob)

refl =  func.reflectivity(vp_model,rhob_model)
trace = np.convolve(refl, wavelet, 'same')

# ----------------------------------------------------------------------
# -------------------ADD RUIDO GAUSSIANO ------------------------
trace_snr_0db = func.gaussian_noise(trace, 0)
trace_snr_5db = func.gaussian_noise(trace, 5)
trace_snr_10db = func.gaussian_noise(trace, 10)
trace_snr_15db = func.gaussian_noise(trace, 15)
trace_snr_20db = func.gaussian_noise(trace, 20)
trace_snr_25db = func.gaussian_noise(trace, 25)

plt.figure(figsize=(3,7))

plt.subplot(161)
plt.title('SNR 0 dB')
plt.plot(trace_snr_0db, t, 'r', label='Noise')
plt.plot(trace, t, 'b', label='Original Trace')
plt.ylabel('Time [s]')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()
plt.ylim(0.5,0.3)

plt.subplot(162)
plt.title('SNR 5 dB')
plt.plot(trace_snr_5db, t, 'r', label='Noise')
plt.plot(trace, t, 'b', label='Original Trace')
plt.ylabel('Time [s]')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()
plt.ylim(0.5,0.3)

plt.subplot(163)
plt.title('SNR 10 dB')
plt.plot(trace_snr_10db, t, 'r', label='Noise')
plt.plot(trace, t, 'b', label='Original Trace')
plt.ylabel('Time [s]')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()
plt.ylim(0.5,0.3)

plt.subplot(164)
plt.title('SNR 15 dB')
plt.plot(trace_snr_15db, t, 'r', label='Noise')
plt.plot(trace, t, 'b', label='Original Trace')
plt.ylabel('Time [s]')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()
plt.ylim(0.5,0.3)

plt.subplot(165)
plt.title('SNR 20 dB')
plt.plot(trace_snr_20db, t, 'r', label='Noise')
plt.plot(trace, t, 'b', label='Original Trace')
plt.ylabel('Time [s]')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()
plt.ylim(0.5,0.3)

plt.subplot(166)
plt.title('SNR 25 dB')
plt.plot(trace_snr_25db, t, 'r', label='Noise')
plt.plot(trace, t, 'b', label='Original Trace')
plt.ylabel('Time [s]')
plt.legend(loc='upper right', fontsize=11)
plt.gca().invert_yaxis()
plt.ylim(0.5,0.3)

plt.tight_layout()
plt.show()



