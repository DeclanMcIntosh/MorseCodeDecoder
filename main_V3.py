from scipy.io import wavfile
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener
import cv2
import numpy as np
import itertools
import collections


# TODO MAKE ACTUALLY WORK MORE BETTER
# FIRST TAKE SLIDING WINDOW FFT FOR MORE RESOLUTION Just move up like 256 (or 512) samples each time then take FFT again for higer time domain resolution lol
# MAKE KERNELS TO CHECK AROUND CURRENT SAMPLE IF IT IS A SHORT OR LONG OR WHATEVER 
# IF SEVERAL SHORTS OR LONGS CO-ENCIDE THATS A SYMBOL
# IF SEVERAL SYMBOLS CO-ENCIDE THATS A CALL SIGN 
# TEST TEST TEST

def fftMag(N, data):
    output = scipy.fftpack.fft(data, n=N)

    output = np.abs(output)
     
    return output[0:N//2]

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

INTS = [
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

CHARS =[
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
    'Z']

#############
FFT_SIZE = 2048
FFT_STEP = 128
display = False

# read file
rate, data = wavfile.read('96kHz-morse.wav')
print(rate)

# get ffts of our single for processing
#complexSignal = np.empty(data.shape[0], np.complex64)
#complexSignal.real, complexSignal.imag = data[:,0], data[:,1]
signal_I = data[:,0]
signal_Q = data[:,1]
output = np.zeros((data.shape[0]//FFT_STEP, FFT_SIZE//2))
for x in range(data.shape[0]//FFT_STEP):
    mags = fftMag(FFT_SIZE, signal_I[x*FFT_STEP:x*FFT_STEP + FFT_SIZE]) + fftMag(FFT_SIZE, signal_Q[x*FFT_STEP:x*FFT_STEP + FFT_SIZE])
    output[x] = mags

#output[output<np.mean(output) + np.std(output)*9] = 0
#output[output>0] = 1

def rebin(arr, downSample0, downSample1):
    shape = (   arr.shape[0]//downSample0, downSample0,
                arr.shape[1]//downSample1, downSample1)
    
    return arr[0:(arr.shape[0]//downSample0)*downSample0,0:(arr.shape[1]//downSample1)*downSample1].reshape(shape).mean(-1).mean(1)


output = wiener(output, (25, 1))

binSize = 4
output = rebin(output,binSize,1)
#print(np.max(output))
#output[output<1] = 0
#output[output>8] = 1

# display fft waterfall type stuff
if display:
    offset = 1000
    img = output[offset:950+offset,:] 
    img = img / np.max(img)
    map_color = cv2.COLORMAP_HOT
    img = np.uint8(img * 255)
    img = cv2.applyColorMap(img,map_color)
    cv2.imshow('window', img)
    cv2.waitKey(-1)

output[output<np.mean(output) + np.std(output)*9] = 0
output[output>0] = 1

if display:
    offset = 1000
    img = output[offset:950+offset,:] 
    img = img / np.max(img)
    map_color = cv2.COLORMAP_HOT
    img = np.uint8(img * 255)
    img = cv2.applyColorMap(img,map_color)
    cv2.imshow('window2', img)
    cv2.waitKey(-1)


# TODO REMOVE fOR TESTING 
#offset = 1000
#output = output[offset:950+offset,:] 

# min negatives between signs =

minSeperation = 40#35 # score 16
minSignalLength = 75#75 # score 16

output_callsigns = []


signalPulses = []

for x in range(1,output.shape[1]):
    currentSignals = []
    currentSymbols = []
    positiveCount = 0
    negativeCount = 100
    #print(x)
    startingPos = 0
    lastPositive = 0
    print(x)
    for y in range(output.shape[0]-1-minSeperation):
        # count past negative and positive pulse sizes
        if np.mean(output[y:y+minSeperation,x]) == 0:
            if y - startingPos > minSignalLength:
                signal = output[startingPos:y+minSeperation//2,x]
                #plt.plot(signal)
                #plt.show()
                signalPulses.append(signal)
            startingPos = y 


output_callsigns = []

def checkValid(symbols):
    valid = True
    # Check valid formatting
    if len(symbols) > 3 and len(symbols) < 7:
        if not currentSymbols[1] in INTS and not currentSymbols[2] in INTS:
            valid = False
        if currentSymbols[1] in INTS and currentSymbols[2] in INTS:
            valid = False
        if currentSymbols[0] in INTS:
            valid = False 
        if currentSymbols[-1] in INTS:
            valid = False
    else:
        valid = False
    return valid

for signal in signalPulses:
    #always starts low
    durations = []
    count = 0
    for x in range(signal.shape[0]-1):
        if signal[x] == signal[x+1]:
            count+=1
        else:
            if signal[x] == 0:
                durations.append(-count-1)  
            if signal[x] != 0:
                durations.append(count+1)  
            count=0
    #print(durations)   

    negativeDurations = []
    positiveDurations = []

    for x in durations:
        if x > 0:
            positiveDurations.append(x)
        else:
            negativeDurations.append(x)

    mean1s = np.mean(np.array(positiveDurations))
    mean0s = np.mean(np.array(negativeDurations))

    coding = ""
    for x in durations:
        if x > 0:
            if x > mean1s:
                coding = coding + "_"
            else:
                coding = coding + "."
        if x < 0:
            if abs(x) > abs(mean0s):
                coding = coding + "/"

    #print(coding)

    temp = list([])
    currentSymbols = list([])
    badValueFlag = False
    for x in coding:
        if x == '.':
            temp.append(0)
        if x == '_':
            temp.append(1)
        if x == '/':
            if len(temp) > 0 and temp in CODES:
                currentSymbols.append(VALUES[CODES.index(temp)])
            elif len(temp) > 0:
                badValueFlag = True
            temp = list([])
    if len(temp) > 0 and temp in CODES:
        currentSymbols.append(VALUES[CODES.index(temp)])
    elif len(temp) > 0:
        badValueFlag = True
    temp = list([])    


    if not badValueFlag and checkValid(currentSymbols):

        print(currentSymbols)
        output_callsigns.append(currentSymbols)


output_callsigns_final = []


print(output_callsigns_final)

print("POST FILTERING")

# process each channel to pick out all signs
for val in output_callsigns:
    output_callsigns_final.append( "".join(val))

#output_callsigns_final = [item for item, count in collections.Counter(output_callsigns_final).items() if count > 1]

output_callsigns_final.sort()
output_callsigns_final = list(output_callsigns_final for output_callsigns_final,_ in itertools.groupby(output_callsigns_final))

print(output_callsigns_final)

print(len(output_callsigns_final))

'''
# TODO TO IMPROVE 
no issues right now with false positives
can add in filtering for things smaller than 1 to be filtered out so the detection of the ends is easier


'''