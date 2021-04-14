import brainflow
import numpy as np
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
import pandas as pd


def main(i):

    board_id = BoardIds.SYNTHETIC_BOARD.value
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    timestamp = BoardShim.get_timestamp_channel(board_id)

    style.use('fivethirtyeight')
    plt.title("Live EEG Datastream from Brainflow", fontsize=15)
    plt.ylabel("Data in millivolts", fontsize=15)
    plt.xlabel("\nTime", fontsize=10)

    data = DataFilter.read_file('data.csv') 
    eegdf = pd.DataFrame(np.transpose(data[eeg_channels, timestamp])) 
    #timedf = pd.DataFrame(np.transpose(data[timestamp])) #to keep it simple, making another dataframe for the timestamps to access later
    
    eegdf_col_names = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
    eegdf.columns = eegdf_col_names

    print("EEG Dataframe")
    print(eegdf) #easy way to check what data is being streamed and if program is working

    #print(eegdf)
    eeg1 = eegdf.iloc[:, 0].values #I am using OpenBCI Ganglion board, so only have four channels.
    eeg2 = eegdf.iloc[:, 1].values 
    eeg3 = eegdf.iloc[:, 2].values  
    eeg4 = eegdf.iloc[:, 3].values
    timex= eegdf.iloc[:, 15].values #timestamps
    #print(timex) #use this to see what the UNIX timestamps look like
    print("EEG Channel 1")
    print(eeg1)
    #print("Time DF")
    #print(timedf)
    print("Timestamp")
    print(timex)

    """plt.cla()
    plt.plot(timex, eeg1, label = "Channel 1", color="red")
    plt.plot(timex, eeg2, label = "Channel 2", color="blue")
    plt.plot(timex, eeg3, label = "Channel 3", color="orange")
    plt.plot(timex, eeg4, label = "Channel 4", color="purple")
    plt.tight_layout()"""


ani = FuncAnimation(plt.gcf(), main, interval=3000)
plt.tight_layout()
#plt.autoscale(enable=True, axis="y", tight=True)
plt.show()