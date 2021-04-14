import pandas as pd
from matplotlib import style
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import brainflow
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams



def main(i):
    BoardShim.enable_dev_board_logger()
    # use synthetic board for demo
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    timestamp = BoardShim.get_timestamp_channel(board_id)
    
    board.prepare_session()
    board.start_stream()
    style.use('fivethirtyeight')
    plt.title("Live EEG stream from Brainflow", fontsize=15)
    plt.ylabel("Data in millivolts", fontsize=15)
    plt.xlabel("\nTime", fontsize=10)
    keep_alive = True

    eeg1 = [] #lists to store eeg data
    eeg2 = []
    eeg3 = []
    eeg4 = []
    timex = [] #list to store timestamp

    while keep_alive == True:

        while board.get_board_data_count() < 250: #ensures that all data shape is the same
            time.sleep(0.005)
        data = board.get_current_board_data(250)

        # creating a dataframe of the eeg data to extract eeg values later
        eegdf = pd.DataFrame(np.transpose(data[eeg_channels]))
        eegdf_col_names = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7",
                           "ch8", "ch9", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16"]
        eegdf.columns = eegdf_col_names

        # to keep it simple, making another dataframe for the timestamps to access later
        timedf = pd.DataFrame(np.transpose(data[timestamp]))

        print("EEG Dataframe") #easy way to check what data is being streamed and if program is working
        print(eegdf)           #isn't neccesary.

        for count, channel in enumerate(eeg_channels):
            # filters work in-place
            # Check Brainflow docs for more filters
            if count == 0:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            if count == 1:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            if count == 2:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            if count == 3:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31

        # Brainflow ML Model
        bands = DataFilter.get_avg_band_powers(
            data, eeg_channels, sampling_rate, True)
        feature_vector = np.concatenate((bands[0], bands[1]))
        print(feature_vector)

        # calc concentration
        concentration_params = BrainFlowModelParams(
            BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        concentration = MLModel(concentration_params)
        concentration.prepare()
        print('Concentration: %f' % concentration.predict(feature_vector))
        concentration.release()

        # calc relaxation
        relaxation_params = BrainFlowModelParams(
            BrainFlowMetrics.RELAXATION.value, BrainFlowClassifiers.KNN.value)
        relaxation = MLModel(relaxation_params)
        relaxation.prepare()
        print('Relaxation: %f' % relaxation.predict(feature_vector))
        relaxation.release()
      
        #appending eeg data to lists 
        eeg1.extend(eegdf.iloc[:, 0].values) # I am using OpenBCI Ganglion board, so I only have four channels.
        eeg2.extend(eegdf.iloc[:, 1].values) # If you have a different board, you should be able to copy paste
        eeg3.extend(eegdf.iloc[:, 2].values) # these commands for more channels.
        eeg4.extend(eegdf.iloc[:, 3].values)
        timex.extend(timedf.iloc[:, 0].values)  # timestamps

        plt.cla()
        #plotting eeg data
        plt.plot(timex, eeg1, label="Channel 1", color="red")
        plt.plot(timex, eeg2, label="Channel 2", color="blue")
        plt.plot(timex, eeg3, label="Channel 3", color="orange")
        plt.plot(timex, eeg4, label="Channel 4", color="purple")
        plt.tight_layout()
        keep_alive = False #resetting stream so that matplotlib can plot data
      

    board.stop_stream()
    board.release_session()


ani = FuncAnimation(plt.gcf(), main, interval=1000) #this essentially calls the function several times until keyboard interrupt
plt.tight_layout()
plt.autoscale(enable=True, axis="y", tight=True)
plt.show()
