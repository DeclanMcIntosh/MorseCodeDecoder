from scipy.io import wavfile
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_fft(N_, data): 
    # Number of samplepoints
    N = N_
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = data[0:1024]
    yf = scipy.fftpack.fft(y, n=N_)
    xf = np.linspace(0, int(1.0/(2.0*T)), int(N/2))

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()


def fftMag(N, data):
    output = scipy.fftpack.fft(data, n=N)

    output = np.abs(output)
     
    return output[0:N//2]


rate, data = wavfile.read('96kHz-morse.wav')

print(rate)

channel_0 = data[:,0]
channel_1 = data[:,1]


FFT_SIZE = 1024

test = fftMag(FFT_SIZE, channel_0[0:1024])

print("")