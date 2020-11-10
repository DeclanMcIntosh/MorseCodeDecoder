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
itterations = 800
map_color = cv2.COLORMAP_HOT
display = True 

# get FFTs
output = np.zeros((itterations,FFT_SIZE//2))
for x in range(itterations):
    mags = fftMag(FFT_SIZE, channel_0[x*FFT_SIZE:(x+1)*FFT_SIZE])
    output[x] = mags

# normalize
output = output / np.max(output)

# clip 
std = np.std(output)

output[output<std*3] = 0
img = output # filered image for display
output[output>=std*3] = 1

# make some filters for the longs and shorts
longsKernels  = np.ones(shape=(1,13))           # len of longs, the threashold should be set to 9 or something
shortsKernels = np.ones(shape=(1,11)) - 2       # 3 * len of shorts 
shortsKernels[:,3:8] = shortsKernels[:,4:9] + 2
# clip again

# generate Iamge
if display:
    img = np.uint8(img * 255)
    img = cv2.applyColorMap(img,map_color)

    cv2.imshow('window', img)
    cv2.waitKey(-1)

print("")