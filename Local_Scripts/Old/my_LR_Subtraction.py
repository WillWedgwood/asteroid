import torch
from asteroid.models import ConvTasNet
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from asteroid import separate

# Load Model
model = ConvTasNet.from_pretrained('mhu-coder/ConvTasNet_Libri1Mix_enhsingle')

method = 'combined_input' #combined_output

if method == 'combined_output':
    # Combined separated outputs
    left_separated, fs = sf.read("Results/Sky_FullFile_Results/Left/Football_Left_2_est1.wav")
    right_separated, fs = sf.read("Results/Sky_FullFile_Results/Right/Football_Right_2_est1.wav")

    combined_out = (left_separated + right_separated) / 2
    sf.write("Combined_out.wav", combined_out, fs)

    # Function to plot spectrogram
    def plot_spectrogram(signal, fs, start_time, end_time):
        # Convert start and end time to sample indices
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        # Extract the segment of interest
        segment = signal[start_sample:end_sample]
        
        # Compute spectrogram
        f, t, Sxx = spectrogram(segment, fs)
        
        # Plot spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        
        # Set log scale for frequency axis
        plt.yscale('log')
        plt.ylim([20, fs / 2])
        
        # Add green dotted lines at 100 Hz and 3000 Hz
        plt.axhline(100, color='green', linestyle='dotted', linewidth=2)
        plt.axhline(3000, color='green', linestyle='dotted', linewidth=2)
        
        # Add red lines at 50 Hz and 8000 Hz
        plt.axhline(50, color='red', linestyle='-', linewidth=2)
        plt.axhline(8000, color='red', linestyle='-', linewidth=2)
        
        # # Highlight strong areas of energy above 5000 Hz
        # high_freq_mask = f > 5000
        # Sxx_high = Sxx[high_freq_mask, :]

        # # Compute the average energy for each time segment in the high-frequency range
        # average_energy = np.mean(10 * np.log10(Sxx_high), axis=0)

        # # Define the threshold for strong energy highlighting
        # energy_threshold = np.percentile(average_energy, 90)  # 98th percentile as threshold

        # # Highlight time segments where the average energy exceeds the threshold
        # for i in range(len(t) - 1):
        #     if average_energy[i] > energy_threshold:
        #         plt.axvspan(t[i], t[i + 1], color='yellow', alpha=0.1)

        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.colorbar(label='Intensity [dB]')
        plt.show()

    # Define the start and end times in seconds
    start_time = 10  # change as needed
    end_time = 20    # change as needed

    # Plot the spectrogram for the specified time segment
    plot_spectrogram(combined_out, fs, start_time, end_time)


if method == 'combined_input':
    # Read Data
    left_data, fs = sf.read("Audio_Sources/Sky/Football/Football_Right_splits/Football_Right_2.wav")
    right_data, fs = sf.read("Audio_Sources/Sky/Football/Football_Left_splits/Football_Left_2.wav")
    noise = left_data - right_data

    combined_in = left_data + right_data / 2

    sf.write("combined_in.wav", combined_in, fs)

    separate.separate(model, "combined_in.wav", 'Results', resample=True, force_overwrite=True)