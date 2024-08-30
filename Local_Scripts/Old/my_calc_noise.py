import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Paths to input and speech audio files
input_audio_path = "Audio_Sources/wolves_commentary_mono_short.wav"
speech_audio_path = "Results/wolves_commentary_mono_short_est1.wav"

# Read the input noisy waveform and the enhanced speech waveform
input_wav, fs_input = sf.read(input_audio_path)
speech_wav, fs_speech = sf.read(speech_audio_path)

# Ensure that the sampling rates match
assert fs_input == fs_speech, "Sampling rates do not match!"

# Ensure the lengths of the signals match
min_length = min(len(input_wav), len(speech_wav))
input_wav = input_wav[:min_length]
speech_wav = speech_wav[:min_length]

# Calculate absolute values of the enhanced speech
abs_speech_wav = np.abs(speech_wav)

# Find indices where the absolute value is greater than 0.1
indices = np.where(abs_speech_wav > 0.1)[0]

# Calculate the average of the absolute values of the enhanced speech at those indices
avg_abs_speech = np.mean(abs_speech_wav[indices])

# Calculate the average of the absolute values of the noisy input at the same indices
avg_abs_input = np.mean(np.abs(input_wav)[indices])

# Compute the ratio of these two averages
ratio = avg_abs_input / avg_abs_speech

# Scale the enhanced speech by this ratio
scaled_speech_wav = speech_wav * ratio

# Calculate noise by subtracting the scaled enhanced speech from the input noisy signal
noise_wav = input_wav - scaled_speech_wav

# Save the noise waveform
sf.write('Results/Noise.wav', noise_wav, fs_input)

# Plotting the waveforms
plt.figure(figsize=(14, 8))

# Plot input noisy waveform
plt.subplot(3, 1, 1)
plt.plot(input_wav, label="Input Noisy Signal")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Input Noisy Signal")

# Plot scaled enhanced speech waveform
plt.subplot(3, 1, 2)
plt.plot(scaled_speech_wav, label="Scaled Enhanced Speech", color='orange')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Scaled Enhanced Speech Signal")

# Plot noise waveform
plt.subplot(3, 1, 3)
plt.plot(noise_wav, label="Noise", color='red')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Noise Signal")

plt.tight_layout()
plt.show()
