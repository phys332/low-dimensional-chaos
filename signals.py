import wave
import numpy as np
import matplotlib.pyplot as plt




def load_audio_signal(output_filename="signal.wav"):
    """
    Loads the .wav file passed in, shoudl typically be used in combination with the above method 
    to record and read out an audio file
    
    output_filename: .wav file to read from
    """
    with wave.open(output_filename, 'rb') as wf:
        # Extract audio parameters
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        # Read frames and convert to numpy array
        audio_data = wf.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # If stereo, reshape array to separate channels
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)

        # audio_array is the non normalized signal

        max_value = np.max(audio_array)
        normalized_audio_array = audio_array / max_value
        normalized_audio_array = np.append(normalized_audio_array, np.zeros(100001-normalized_audio_array.shape[0]))
        
    return normalized_audio_array


def square_wave_signal(length=5):
    """
    Generates a random binary signal
    Length: # of nodes in the signal (0 or 1)
    """
    A = 1       # amplitude
    Tp = 1      # pulse duration

    signal = np.zeros(length)
    for index in range(length):
        signal[index] = np.round(np.random.rand())

    return signal


'''Record generic audio signal (voice or song)'''
def main():
    return
    

if __name__ == "__main__":
    main()
    