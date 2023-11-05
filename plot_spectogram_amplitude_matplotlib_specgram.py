import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import importlib
import config
import numpy as np
from scipy.fftpack import fft, ifft,fftfreq,fftshift
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import
from matplotlib import pyplot as plt
import scipy.io
from CSIKit.reader import get_reader


my_samples = []
my_toa = []

pcap_filename = 'D:/Data/fine_grained/Classroom/80MHz/3mo/m1/CSI_pcap/T.pcap'#input('Pcap file name: ')

if '.pcap' not in pcap_filename:
    pcap_filename += '.pcap'
pcap_filepath = pcap_filename#'/'.join([config.pcap_fileroot, pcap_filename])

try:
    samples = decoder.read_pcap(pcap_filepath)
    reader = get_reader(pcap_filepath)
    csi_data = reader.read_file(pcap_filepath, scaled=False)
    try:
        x = csi_data.timestamps
        x = [timestamp-x[0] for timestamp in x]
        print(len(x))
    except AttributeError as e:
            #No timestamp in frame. Likely an IWL entry.
            #Will be moving timestamps to CSIData to account for this.
        x = [0]
        print("No timestamp in frame.")
        # if sum(x) == 0:
        #     #Some files have invalid timestamp_low values which means we can't plot based on timestamps.
        #     #Instead we'll just plot by frame count.

        #     xlim = no_frames

        #     x_label = "Frame No."
        # else:
        #     xlim = max(x)

        # limits = [0, xlim, 1, no_subcarriers]
    
except FileNotFoundError:
    print(f'File {pcap_filepath} not found.')
    exit(-1)

try:
    i=0
    while (i<samples.nsamples):
        csi = samples.get_csi(
                    i,
                    config.remove_null_subcarriers,
                    config.remove_pilot_subcarriers
                )
        my_samples.append(csi)
        #my_toa.append(samples.)
        i+=1
except:
    print("Problem")        

######################################
#regular fft plot
my_samps = np.asarray(my_samples)
data = pd.DataFrame(my_samps)


print(data.shape)

INDEX = 1

for i in range(0, 128):

    INDEX = i+1

    cols1 = data.iloc[:, i].to_numpy().flatten()
    #the columns are the subcarriers with complex numbers
    csi = cols1
    print(csi.shape)
    # Compute the spectrogram using plt.specgram
    fig, ax = plt.subplots(figsize=(10, 5))
    Fs = 1000
    Pxx, freqs, bins, im = ax.specgram(np.abs(csi-np.mean(csi)), NFFT=256, Fs=Fs, noverlap=200, cmap='jet', mode="magnitude", sides="twosided", scale_by_freq=True)
    
    # Convert PSD values to dB
    Pxx_dB = 10 * np.log10(Pxx)

    # Set the display range in dB
    vmin = Pxx_dB.min() #- 100
    vmax = Pxx_dB.max()

    # Display the spectrogram in dB
    im.set_clim(30, 100)
    ref_val = 7000
    actual_val = csi.shape[0]
    y_val = (ref_val/actual_val)*(Fs/2)
    #ax.set_ylim(int(-y_val), int(y_val))
    
    # Set the x and y axis labels
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    # Show the colorbar
    fig.colorbar(im).set_label('Magnitude [dB]')
    plt.title('CSI Amplitude Spectogram for subcarrier #'+str(INDEX))
    #plt.savefig("C:/Users/ajaya/OneDrive - Mississippi State University/Desktop/nexmon_csi-feature-python/utils/python/My Dataset/Plots/0703_APT_Data_Collection/TPLINK/Running/"+"withSobo0703_sameLOC_TPLINK_running_Subcarrier_"+str(INDEX)+".png")
    # Show the plot
    plt.show()

