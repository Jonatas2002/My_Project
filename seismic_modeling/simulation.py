from modeling import scalar
from time import time

def simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

   # myWave[id].get_type()

    myWave[id].set_wavelet()
    myWave[id].set_model()

    
    
    start = time()
    myWave[id].wave_propagation()
    end = time()

    print(end - start)
    
    myWave[id].plot_wavelet()
    myWave[id].plot_model()
    myWave[id].plot_wavefield()
    myWave[id].plot_seismogram()
    myWave[id].plot_wave_propagation()

if __name__ == "__main__":
    simulation()