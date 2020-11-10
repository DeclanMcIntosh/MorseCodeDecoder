from scipy.io import wavfile
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools

# 1 is long 0 is short
CODES = [
    [0,1]       ,
    [1,0,0,0]   ,
    [1,0,1,0]   ,
    [1,0,0]     ,
    [0]         ,
    [0,0,1,0]   ,
    [1,1,0]     ,
    [0,0,0,0]   ,
    [0,0]       ,
    [0,1,1,1]   ,
    [1,0,1]     ,
    [0,1,0,0]   ,
    [1,1]       ,
    [1,0]       ,
    [1,1,1]     ,
    [0,1,1,0]   ,
    [1,1,0,1]   ,
    [0,1,0]     ,
    [0,0,0]     ,
    [1]         ,
    [0,0,1]     ,
    [0,0,0,1]   ,
    [0,1,1]     ,
    [1,0,0,1]   ,
    [1,0,1,1]   ,
    [1,1,0,0]   ,
    [0,1,1,1,1] ,
    [0,0,1,1,1] ,
    [0,0,0,1,1] ,
    [0,0,0,0,1] ,
    [0,0,0,0,0] ,
    [1,0,0,0,0] ,
    [1,1,0,0,0] ,
    [1,1,1,0,0] ,
    [1,1,1,1,0] ,
    [1,1,1,1,1] ]

VALUES =[
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '0']


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

# get magnitude of FFT of some set of data
def fftMag(N, data):
    output = scipy.fftpack.fft(data, n=N)

    output = np.abs(output)
     
    return output[0:N//2]


def process_channel(data_channel, itterations):
        
    FFT_SIZE = 1024*2
    itterations = itterations

    print(data_channel.shape[0]/(FFT_SIZE*itterations))
    map_color = cv2.COLORMAP_HOT
    display = False 

    # get FFTs
    output = np.zeros((itterations,FFT_SIZE//2))
    for x in range(itterations):
        mags = fftMag(FFT_SIZE, data_channel[x*FFT_SIZE:(x+1)*FFT_SIZE])
        #if (x+1) * FFT_SIZE > data_channel.shape[0]:
        #    print("TOO BIG")
        #    print((x+1) * FFT_SIZE)
        output[x] = mags

    # normalize
    output = output / np.max(output)

    # clip 
    std = np.std(output)

    output[output<std*10] = 0
    img = output # filered image for display
    #output[output>=std*6] = 1

    #       # make some filters for the longs and shorts
    #       longsKernels  = np.ones(shape=(1,13))           # len of longs, the threashold should be set to 9 or something
    #       shortsKernels = np.ones(shape=(1,9)) - 2       # 3 * len of shorts 
    #       shortsKernels[:,2:7] = shortsKernels[:,2:7] + 2
    #       # clip again

    MAX_SPACES_BETWEEN_CHARECTERS = 7
    MAX_SPACES_BETWEEN_PULSES = 3
    MAX_POSITIVES_AS_SHORT = 3

    allDetectedCallSigns = [] # time, sign

    for x in range(FFT_SIZE//2):
        current_signs = []
        current_morse = [] 
        positiveCount = 0
        negativeCount = 0 
        for y in range(itterations):
            #wait untill we find a non-zero value
            if output[y,x] > 0:
                if negativeCount > MAX_SPACES_BETWEEN_PULSES:
                    if current_morse in CODES:
                        current_signs.append(VALUES[CODES.index(current_morse)])
                        #print(current_morse)
                    current_morse = []
                if negativeCount > MAX_SPACES_BETWEEN_CHARECTERS:
                    if len(current_signs) in [4,5,6]:
                        # filter out call signs that are just shorts, this is generally due to noise
                        filterSet = [K for K in current_signs if K not in ['H','E','I','S','5','T']]
                        if not len(filterSet) == 0:
                            outputName = ""
                            for sign in current_signs:
                                outputName = outputName + sign
                            #print(outputName)
                            # get the time in seconds from the start of the file based on index
                            time = int(y*FFT_SIZE/96000)
                            print( "Timestamp in seconds: " + str(time) + "   Callsign = "+ outputName)
                            allDetectedCallSigns.append([time, outputName])
                    current_signs = []
                    current_morse = []

                # handel tracked values
                positiveCount += 1
                negativeCount =  0 
            else:
                if positiveCount > MAX_POSITIVES_AS_SHORT: # it is a long
                    current_morse.append(1)
                elif positiveCount > 0 and positiveCount <= MAX_POSITIVES_AS_SHORT:
                    current_morse.append(0)

                # handel tracked values
                negativeCount += 1
                positiveCount =  0

    # get rids of duplicates that show up in multiple frequency bins
    allDetectedCallSigns.sort()
    list(allDetectedCallSigns for allDetectedCallSigns,_ in itertools.groupby(allDetectedCallSigns))
    print(allDetectedCallSigns)
     

    # generate Iamge
    if display:
        img = np.uint8(img * 255)
        img = cv2.applyColorMap(img,map_color)

        cv2.imshow('window', img)
        cv2.waitKey(-1)

    print("")

    return allDetectedCallSigns

rate, data = wavfile.read('96kHz-morse.wav')

print(rate)

channel_0 = data[:,0]
channel_1 = data[:,1]

# we repeate the proccess with offsets to capture things on transitions of FFTs and on both IQ channels
allDetectedCallSigns = process_channel(channel_0, 27840*6)
allDetectedCallSigns2 = process_channel(channel_0[512:], 27840*6)
allDetectedCallSigns1 = process_channel(channel_1, 27840*6)
allDetectedCallSigns3 = process_channel(channel_1[512:], 27840*6)

allDetectedCallSigns = allDetectedCallSigns + allDetectedCallSigns1 + allDetectedCallSigns2 + allDetectedCallSigns3

allDetectedCallSigns.sort()
list(allDetectedCallSigns for allDetectedCallSigns,_ in itertools.groupby(allDetectedCallSigns))
print(allDetectedCallSigns)
     
f = open("outputCallSigns.txt", "a")

for stuff in allDetectedCallSigns:
    f.write(str(stuff[0]) + str(stuff[1]))

f.close
