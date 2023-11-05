import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
#from scapy.all import rdpcap
from scipy.signal import spectrogram


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


# Step 1: Find the Wi-Fi_Nexmon path
def find_wifi_nexmon_path(root_dir):
    wifi_nexmon_paths = []
    for root, dirs, files in os.walk(root_dir):
        if "Wi-Fi_Nexmon" in dirs:
            wifi_nexmon_paths.append(os.path.join(root, "Wi-Fi_Nexmon"))
    print("There were "+ str(len(wifi_nexmon_paths))+ " paths found!")
    return wifi_nexmon_paths

# Step 2: Create a new folder named "Spectrogram"
def create_spectrogram_folder(wifi_nexmon_paths):
    spectrogram_dir_col = []
    for i in wifi_nexmon_paths:
        spectrogram_dir = os.path.join(i, "Spectrogram")
        os.makedirs(spectrogram_dir, exist_ok=True)
        spectrogram_dir_col.append(spectrogram_dir)
    print("Spectogram folder created sucessfully in each Wi-Fi_Nexmon Directory")
    return spectrogram_dir_col

# Step 3: Get paths for each file in the directory
def get_pcap_files(directory):
    pcap_files = []
    for dir in directory:
        # pcap_files.append(f for f in os.listdir(dir) if f.endswith(".pcap"))
        for f in os.listdir(dir):
            if f.endswith(".pcap"):
                pcap_files.append(os.path.join(dir, f))
    #print(pcap_files)
    print("There were "+ str(len(pcap_files))+" pcap files found!")
    return pcap_files
    #return [os.path.join(directory, file) for file in pcap_files]

# Step 4: Generate spectrograms for subcarriers #6-22
def generate_spectrograms(pcap_file, output_dir):
    packets = rdpcap(pcap_file)
    subcarrier_data = []
    
    for packet in packets:
        if "Dot11Beacon" in packet and hasattr(packet, "subcarriers"):
            subcarrier_data.append(packet.subcarriers[6:23])

    for i, subcarriers in enumerate(subcarrier_data):
        _, _, Sxx, _ = spectrogram(subcarriers, fs=20e6, nperseg=256, noverlap=128, mode='complex')
        Sxx_db = 10 * np.log10(np.abs(Sxx))
        
        # Save the spectrogram as an image
        plt.figure(figsize=(2, 2))
        plt.imshow(Sxx_db, aspect='auto', cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"spectrogram_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

# Step 5: Combine 16 images into a 4x4 grid
def combine_images(images, output_path):
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                img = plt.imread(images[index])
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    root_directory = "D:/Data/coarse/Classroom/80MHz/3mo/m1/CSI_pcap/"
    print(root_directory)
    # Step 1: Find the Wi-Fi_Nexmon path
    wifi_nexmon_path = find_wifi_nexmon_path(root_directory)
    print(wifi_nexmon_path)
    # Step 2: Create a new folder named "Spectrogram"
    spectrogram_directory = create_spectrogram_folder(wifi_nexmon_path)

    # Step 3: Get paths for each file in the directory
    pcap_files_all = get_pcap_files(wifi_nexmon_path)

    start_index = 0
    step = 1

    # Calculate the indices to select based on the pattern
    indices_to_select = [start_index + i * step for i in range(len(pcap_files_all)) if start_index + i * step < len(pcap_files_all)]

    # Create a new list by selecting elements at specified indices
    pcap_files = [pcap_files_all[i] for i in indices_to_select]
    
    print(len(pcap_files))
    print("______________")
#   raise Exception("Stopping the code here")
    # Step 4: Generate spectrograms for subcarriers #6-22
   
    for pcap_file in pcap_files:
        print("pcap: "+pcap_file)
        my_samples = []
        try:
            samples = decoder.read_pcap(pcap_file)
            reader = get_reader(pcap_file)
            csi_data = reader.read_file(pcap_file, scaled=False)
            try:
                x = csi_data.timestamps
                x = [timestamp-x[0] for timestamp in x]
                #print(len(x))
            except AttributeError as e:
                    #No timestamp in frame. Likely an IWL entry.
                    #Will be moving timestamps to CSIData to account for this.
                x = [0]
                print("No timestamp in frame.")
            
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

        my_samps = np.asarray(my_samples)
        data = pd.DataFrame(my_samps)
        print(data.shape)
        INDEX = 0

        for i in range(90, 121):

            INDEX = i

            cols1 = data.iloc[:, i].to_numpy().flatten()
            #the columns are the subcarriers with complex numbers
            csi = cols1
            print(csi.shape)
            # Compute the spectrogram using plt.specgram
            fig, ax = plt.subplots(figsize=(10, 5))
            Fs = 5000
            Pxx, freqs, bins, im = ax.specgram(np.abs(csi-np.mean(csi)), NFFT=256, Fs=Fs, noverlap=240, cmap='jet', mode="magnitude", sides="twosided", scale_by_freq=True)
            print(Pxx.shape)
            print("_____")
            # Convert PSD values to dB
            Pxx_dB = 10 * np.log10(Pxx)

            # Set the display range in dB
            vmin = Pxx_dB.min() #- 100
            vmax = Pxx_dB.max()

            # Display the spectrogram in dB
            im.set_clim(10, 50)
            ref_val = 7000
            actual_val = csi.shape[0]
            y_val = (ref_val/actual_val)*(Fs/2)
            #ax.set_ylim(int(-y_val), int(y_val))
            ax.set_ylim(-Fs/5, Fs/5)
            ax.axis("off")
            ax.margins(0,0)
            # Set the x and y axis labels
            #ax.set_xlabel('Time [s]')
            #ax.set_ylabel('Frequency [Hz]')

            # Show the colorbar
            #fig.colorbar(im).set_label('Magnitude [dB]')
            #plt.title('CSI Amplitude Spectogram for subcarrier #'+str(INDEX))
            file_name = os.path.basename(pcap_file)
            figFilename = os.path.join(os.path.dirname(pcap_file), "Spectrogram")

            #completefigFileName = figFilename +'\\'+file_name+"_INDEX#"+str(INDEX)+".png"
            # print("Saving file: "+ ".\\NewDataset90-120\\"+file_name+"_INDEX#"+str(INDEX)+".png")
            # if not os.path.exists(".\\NewDataset90-120\\"+file_name):
            #     os.makedirs(".\\NewDataset90-120\\"+file_name)
            # plt.savefig(".\\NewDataset90-120\\"+file_name+"\\"+file_name+"_INDEX#"+str(INDEX)+".png", bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close(fig)
        
        #generate_spectrograms(pcap_file, spectrogram_directory)

    
    
    
    # Step 5: Combine 16 images into a 4x4 grid
        #spectrogram_images = sorted(os.listdir(spectrogram_directory))
        #combined_output_path = os.path.join(spectrogram_directory, "combined_spectrograms.png")
        #combine_images([os.path.join(spectrogram_directory, img) for img in spectrogram_images], combined_output_path)
    # else:
    #     print("Wi-Fi_Nexmon directory not found.")
