from numpy import array, diff, where, split
from scipy import arange
import soundfile
import numpy, scipy
import pylab
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')

def findPeak(magnitude_values, noise_level=2000):

    splitter = 0
    # zero out low values in the magnitude array to remove noise
    magnitude_values = numpy.asarray(magnitude_values)

    low_values_indices = magnitude_values < noise_level  # Where values are low
    magnitude_values[low_values_indices] = 0  # Sets the values below the signal threshold outside of the graphical range

    indices = []

    flag_start_looking = False

    both_ends_indices = []

    length = len(magnitude_values)
    for i in range(length):
        if magnitude_values[i].any() != splitter:
            if not flag_start_looking:
                flag_start_looking = True
                both_ends_indices = [0, 0]
                both_ends_indices[0] = i
        else:
            if flag_start_looking:
                flag_start_looking = False
                both_ends_indices[1] = i
                # add both_ends_indices in to indices
                indices.append(both_ends_indices)

    return indices

def extractFrequency(indices, freq_threshold=2):

    extracted_freqs = []

    for index in indices:
        freqs_range = freq_bins[index[0]: index[1]]
        avg_freq = round(numpy.average(freqs_range))

        if avg_freq not in extracted_freqs:
            extracted_freqs.append(avg_freq)

    # group extracted frequency by nearby=freq_threshold (tolerate gaps=freq_threshold)
    group_similar_values = split(extracted_freqs, where(diff(extracted_freqs) > freq_threshold)[0]+1)

    # calculate the average of similar value
    extracted_freqs = []
    for group in group_similar_values:
        extracted_freqs.append(round(numpy.average(group)))

    print("Frequency Components: ", extracted_freqs)
    return extracted_freqs

def ratio(a, b):
    a = float(a)
    b = float(b)
    if b == 0:
        return a
    return ratio(b, a % b)



# def ratio_list(frequencies):
#     for i in frequencies:
#         ratio_list.append(ratio(frequencies[0],i))
#     return ratio_list




if __name__ == '__main__':

    file_path = input('Enter the filename: ')+'.wav'
    # file_path = 'untitled.wav'
    print('Open audio file path: ', file_path)
    print('')
    print('')



    audio_samples, sample_rate  = soundfile.read(file_path, dtype='int16')
    number_samples = len(audio_samples)
    print('SAMPLE INFORMATION')
    print('____________________')
    print('Audio Samples: ', audio_samples)
    print('Number of Samples: ', number_samples)
    print('Sample Rate: ', sample_rate)

    # duration of the audio file
    duration = round(number_samples/sample_rate, 2)
    print('Audio Duration: {0}s'.format(duration))
    print('')

    # list of possible frequencies bins
    freq_bins = arange(number_samples) * sample_rate/number_samples
    print('FREQUENCIES')
    print('____________________')
    print('Frequency Length: ', len(freq_bins))
    print('Frequency bins: ', freq_bins)
    print('')

#     # FFT calculation
    fft_data = scipy.fft(audio_samples)
    print('FFT INFORMATION')
    print('____________________')
    print('FFT Length: ', len(fft_data))
    print('FFT data: ', fft_data)

    freq_bins = freq_bins[range(number_samples//2)]
    normalization_data = fft_data/number_samples
    magnitude_values = normalization_data[range(len(fft_data)//2)]
    magnitude_values = numpy.abs(magnitude_values)

    indices = findPeak(magnitude_values=magnitude_values, noise_level=2000)
    print('Indices: ', indices)
    frequencies = extractFrequency(indices=indices)
    print('')
    print('')
    print("Frequencies: ", frequencies)
    ratiolist = [x / frequencies[0] for x in frequencies]
    print('Frequency Ratios: ', ratiolist)
    print('')
    print('')









# # Testing plotting frequencies to graph

    x_axis_data = freq_bins
    y_axis_data = magnitude_values

    plt.plot(x_axis_data, y_axis_data, color='turquoise') # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Amplitude')

    # # Sets the graph to a logarythmic scale
    plt.xscale('log')

    plt.xlim(20, 20000)
    plt.ylim(bottom=1)
    input('Press any key to view graph...')
    plt.show()
