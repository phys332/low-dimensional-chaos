import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

# will need pyaudio installed to work -- was planning on installing and uninstalling it after every use

def record_audio_signal(output_filename="signal.wav"):
    """
    Function for recording an audio signal for use

    output_filename: file name to be used when saving the signal recorded
    """
    # Parameters for recording
    duration = 5  # seconds
    sample_rate = 44100  # sample rate in Hz
    channels = 1  # not stereo
    chunk = 1024  # size of each buffer chunk

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream for recording
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")

    # Record data in chunks
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recorded data to a .wav file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as '{output_filename}'.")
    return output_filename

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



if __name__ == "__main__":
    length = 10
    # signal = square_wave_signal(10)
    #record_audio_signal()
    signal = load_audio_signal()
    xs = np.arange(1, length + 1, 1)

    
    # plot the audio signal
    # -----------------
    # plt.plot(signal)
    # plt.show()


    print(signal)