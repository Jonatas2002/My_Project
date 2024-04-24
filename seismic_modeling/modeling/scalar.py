import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as movie

from numba import njit, prange  

class Wavefield_1D():
    
    wave_type = "1D wave propagation in constant density acoustic isotropic media"
    
    def __init__(self):
        
        # TODO: read parameters from a file
        
        self.nt = 10001
        self.dt = 1e-3
        self.fmax = 30.0
        
        self.nz = 1001
        self.dz = 10
        self.model = np.zeros(self.nz)
        self.depth = np.arange(self.nz) * self.dz
        self.times = np.arange(self.nt) * self.dt

        
        self.prof = np.array([2000, 4000, 6000, 8000, 10010])
        self.velocities = np.array([1500, 4000, 2000, 4500, 5000])
        
        self.fonte = np.array([100, 500, 1000])
        self.receptor = np.array([3000, 5000, 7000])
        
        self.src_projection = np.array(self.fonte / self.dz, dtype = int)
        self.rec_projection = np.array(self.receptor / self.dz, dtype = int)
        
    def get_type(self):
        print(self.wave_type)
        
    def set_model(self):
        interfaces = []
        for i in range(len(self.prof)):
            start_depth = int(self.prof[i - 1] / self.dz)  if i > 0 else 0
            end_depth = self.prof[i] / self.dz
            vel = self.velocities[i]
            interfaces.append((start_depth, end_depth, vel))

        for j in range(len(interfaces)):
            start, end, vel = interfaces[j]
            start_depth = int(start)
            end_depth = int(end)
            self.model[start_depth:end_depth] = vel
            
    def set_wavelet(self):    
        
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0
        
        arg = np.pi*(np.pi*fc*td)**2.0
        
        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)
        
    def wave_propagation(self):

        self.P = np.zeros((self.nz, self.nt)) # P_{i,n}

        sId = int(self.fonte[0] / self.dz)

        for n in range(1,self.nt-1):

            self.P[sId,n] += self.wavelet[n]    

            laplacian = get_laplacian_1D(self.P, self.dz, self.nz, n)

            self.P[:,n+1] = (self.dt*self.model)**2 * laplacian + 2.0*self.P[:,n] - self.P[:,n-1]
        
        self.P *= 1.0 / np.max(self.P) 

        self.seismogram = self.P[self.rec_projection,:].T 
            
    def plot_wavefield(self):
        fig, ax = plt.subplots(num = "Wavefield plot", figsize = (8, 8), clear = True)

        ax.imshow(self.P, aspect = "auto", cmap = "Greys")

        # ax.plot(self.P[:,5000])

        ax.set_title("Wavefield", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        
        fig.tight_layout()
        fig.savefig('Wavefield.png')
        plt.show() 

    def plot_model(self):
        fig, ax = plt.subplots(figsize=(3, 6), clear=True)
        ax.plot(self.model, self.depth)
        ax.plot(self.model[self.fonte //self.dz], self.fonte,  '*', color='red', label='Fonte', markersize = 10)
        ax.plot(self.model[self.receptor//self.dz], self.receptor,   'v' , color='blue', label='Receptor', markersize = 10)
        ax.set_title("Velocity Model", fontsize=18)
        ax.set_xlabel("Velocity [m/s]", fontsize=15)
        ax.set_ylabel("Depth [m]", fontsize=15) 
        ax.set_ylim(max(self.depth), min(self.depth))
        
        fig.tight_layout()
        fig.savefig('VelocityModel.png')
        plt.show()   
               
            
    def plot_wavelet(self):      
         
        t = np.arange(self.nt)*self.dt
        
        fig, ax = plt.subplots(figsize = (10, 5), clear = True)
        
        ax.plot(t, self.wavelet)
        ax.set_title("Wavelet", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Amplitude", fontsize = 15)    
             
        ax.set_xlim([0, np.max(t)])        
        
        fig.tight_layout()
        plt.show()
        
        
        
    def plot_seismogram(self):

        fig, ax = plt.subplots(num = "Seismogram plot", figsize = (4, 8), clear = True)

        ax.plot(self.seismogram, self.times)
        
        ax.set_title("Seismogram", fontsize = 18)
        ax.set_xlabel("Normalized Amplitude", fontsize = 15)
        ax.set_ylabel("Time [s]", fontsize = 15) 
        
        ax.set_ylim([0, (self.nt-1)*self.dt])
        
        ax.invert_yaxis()
        fig.tight_layout()
        plt.show()

    def plot_wave_propagation(self):

        fig, ax = plt.subplots(num = "Wave animation", figsize = (4, 8), clear = True)

        ax.set_title("Wave propagation", fontsize = 18)
        ax.set_ylabel("Depth [m]", fontsize = 15)
        ax.set_xlabel("Amplitude", fontsize = 15) 
        
        ax.set_ylim([0, (self.nz-1)*self.dz])
        ax.set_xlim([np.min(self.P)-0.5, np.max(self.P)+0.5])

        wave, = ax.plot(self.P[:,0], self.depth)
        srcs, = ax.plot(self.P[self.src_projection,0], self.fonte, "*", color = "black", label = "Sources")
        recs, = ax.plot(self.P[self.rec_projection,0], self.receptor, "v", color = "green", label = "Receivers")

        artists_list = []

        artists_list.append(wave)
        artists_list.append(srcs)
        artists_list.append(recs)

        def init():
            wave.set_ydata(self.depth)  
            wave.set_xdata([np.nan] * self.nz)
                        
            return artists_list

        def animate(i): 
            wave.set_ydata(self.depth)  
            wave.set_xdata(self.P[:,i])
        
            srcs.set_xdata(self.P[self.src_projection,i])
            recs.set_xdata(self.P[self.rec_projection,i]) 

            return artists_list

        ax.legend(loc = "upper right")
        ax.invert_yaxis()
        fig.tight_layout()

        _= movie.FuncAnimation(fig, animate, init_func = init, interval = 1e-3*self.dt, frames = self.nt, blit = True, repeat = True)
        
        plt.show()
        
        
        
        
        
@njit 
def get_laplacian_1D(P, dz, nz, time_id):

    d2P_dz2 = np.zeros(nz)

    for i in prange(1, nz-1): 
        d2P_dz2[i] = (P[i-1,time_id] - 2.0*P[i,time_id] + P[i+1,time_id]) / dz**2.0    

    return d2P_dz2

class Wavefield_2D(Wavefield_1D):
    
    def __init__(self):
        super().__init__()
        
        wave_type = "2D wave propagation in constant density acoustic isotropic media"    


class Wavefield_3D(Wavefield_2D):
    
    def __init__(self):
        super().__init__()
        
        wave_type = "3D wave propagation in constant density acoustic isotropic media"    
