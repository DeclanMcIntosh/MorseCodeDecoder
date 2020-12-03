from scipy.io import wavfile
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools


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
FFT_STEP = 256
display = True

min_period_between_pulse = 2*2#
max_period_between_pulse = 12

min_period_between_symbol = 12*2#
max_period_between_symbol = 35*2#

min_period_short_pulse = 4
max_period_short_pulse = 10

min_period_long_pulse = 10
max_period_long_pulse = 35

short_long_deliminator = 10*2#

# Short is 5-12 samples long 
# Long is 20-30 samples long
# 4-10 samples between shorts or longs
# 15-25 samples long for time between symbols
# longer than that is a new message
############

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

output[output<np.mean(output) + np.std(output)*10] = 0



# display fft waterfall type stuff
if display:
    offset = 2*1000
    img = output[offset:950*2+offset,:] 
    img = img / np.max(img)
    map_color = cv2.COLORMAP_HOT
    img = np.uint8(img * 255)
    img = cv2.applyColorMap(img,map_color)
    cv2.imshow('window', img)
    cv2.waitKey(-1)


# TODO REMOVE fOR TESTING 
offset = 2*1000
output = output[offset:950*2+offset,:] 

# process each channel to pick out all signs

output_callsigns = []

def checkLengthOfPulse(positiveCount):
    if positiveCount < short_long_deliminator:
        return 0
    else:
        return 1

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



for x in range(output.shape[1]):
    currentSignals = []
    currentSymbols = []
    positiveCount = 0
    negativeCount = 100
    #print(x)
    for y in range(output.shape[0]-1):
        # count past negative and positive pulse sizes
        if output[y,x] > 0:
            positiveCount += 1
        else:
            negativeCount += 1

        # on a falling edge of pulse
        if output[y,x] > 0 and not output[y+1,x] > 0:
            
            if positiveCount > min_period_between_pulse:

                if negativeCount > max_period_between_symbol:
                    if checkValid(currentSymbols):
                        output_callsigns.append(currentSymbols)
                        print(currentSymbols)
                        currentSignals = []
                        currentSymbols = []
                
                if negativeCount <=  max_period_between_symbol and negativeCount > min_period_between_symbol:
                    if currentSignals in CODES:
                        currentSymbols.append(VALUES[CODES.index(currentSignals)])
                        currentSignals = []

                currentSignals.append(checkLengthOfPulse(positiveCount))

                negativeCount = 0
                positiveCount = 0  

            # pulse is too short might as well be negative treat it as such         
            else:
                # too shot might as well be negative
                negativeCount = positiveCount + negativeCount
                positiveCount = 0

    if currentSignals in CODES:
        currentSymbols.append(VALUES[CODES.index(currentSignals)])
    if checkValid(currentSymbols):
        output_callsigns.append(currentSymbols)
        print(currentSymbols)
        currentSignals = []
        currentSymbols = []


output_callsigns_final = []
for val in output_callsigns:
    output_callsigns_final.append( "".join(val))

output_callsigns_final.sort()
output_callsigns_final = list(output_callsigns_final for output_callsigns_final,_ in itertools.groupby(output_callsigns_final))
