import time
import brainflow
import numpy as np
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
import pandas as pd


class DataThread (threading.Thread):

    def __init__(self, board, board_id):
        threading.Thread.__init__(self)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.timestamp = BoardShim.get_timestamp_channel(board_id)
        self.keep_alive = True
        self.board = board

    def run(self):
        #
        # style.use('fivethirtyeight')

        while self.keep_alive:
            # get board data removes data from the buffer
            while self.board.get_board_data_count() < 250:
                time.sleep(0.005)
            data = self.board.get_board_data()
            timedf = pd.DataFrame(np.transpose(data[self.timestamp]))
            eegdf = pd.DataFrame(np.transpose(data[self.eeg_channels]))
            eegdf_col_names = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
            eegdf.columns = eegdf_col_names
            #print("EEG 1")
            #print(eeg1)

            print("EEG Dataframe")
            print(eegdf)

            """print("Time Dataframe")
            print(timedf)"""

            for count, channel in enumerate(self.eeg_channels):
                # filters work in-place
                if count == 0:
                    DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                    DataFilter.perform_bandpass(data[channel], self.sampling_rate, 21.0, 20.0, 4,
                                                FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
                if count == 1:
                    DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                    DataFilter.perform_bandpass(data[channel], self.sampling_rate, 21.0, 20.0, 4,
                                                FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
                if count == 2:
                    DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                    DataFilter.perform_bandpass(data[channel], self.sampling_rate, 21.0, 20.0, 4,
                                                FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
                if count == 3:
                    DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                    DataFilter.perform_bandpass(data[channel], self.sampling_rate, 21.0, 20.0, 4,
                                                FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            """eeg1.append(eegdf.iloc[:, 0])
            eeg2.append(eegdf.iloc[:, 1])
            eeg3.append(eegdf.iloc[:, 2])
            eeg4.append(eegdf.iloc[:, 3])
            timex.append(timedf.iloc[:, 0])"""

            """#plt.autoscale(enable=True, axis="y", tight=True)

            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(1, 1, 1)

            plt.title("Neurofeedback", fontsize=15)
            plt.ylabel("Data in UV", fontsize=15)
            plt.xlabel(
                "\nTime", fontsize=10)
            fig.show()

            ani = animation.FuncAnimation(fig, DataThread.run(), interval=1000)
            ax1.plot(timex, eeg1, color="r")
            plt.show()"""
            timex = timedf.iloc[:,0]  # list to store timestamps

            eeg1 = eegdf.iloc[:, 0]  # list to store eeg data in seperate channels
            eeg2 = eegdf.iloc[:, 1]
            eeg3 = eegdf.iloc[:, 2]
            eeg4 = eegdf.iloc[:, 3]
            
            plt.cla()

            plt.plot(timex, eeg1, label = "Channel 1")
            plt.plot(timex, eeg2, label = "Channel 2")
            plt.plot(timex, eeg3, label = "Channel 3")
            plt.plot(timex, eeg4, label = "Channel 4")

            plt.tight_layout()

            """return eeg1
            return eeg2
            return eeg3
            return eeg4
            return timex"""


                    




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

    # style.use('fivethirtyeight')

    while True:
        # get board data removes data from the buffer
        while board.get_board_data_count() < 250:
            time.sleep(0.005)
        data = board.get_board_data()
        timedf = pd.DataFrame(np.transpose(data[timestamp]))
        eegdf = pd.DataFrame(np.transpose(data[eeg_channels]))
        eegdf_col_names = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        eegdf.columns = eegdf_col_names
        #print("EEG 1")
        #print(eeg1)

        print("EEG Dataframe")
        print(eegdf)

        """print("Time Dataframe")
        print(timedf)"""

        for count, channel in enumerate(eeg_channels):
            # filters work in-place
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
        """eeg1.append(eegdf.iloc[:, 0])
        eeg2.append(eegdf.iloc[:, 1])
        eeg3.append(eegdf.iloc[:, 2])
        eeg4.append(eegdf.iloc[:, 3])
        timex.append(timedf.iloc[:, 0])"""

        """#plt.autoscale(enable=True, axis="y", tight=True)

        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        plt.title("Neurofeedback", fontsize=15)
        plt.ylabel("Data in UV", fontsize=15)
        plt.xlabel(
            "\nTime", fontsize=10)
        fig.show()

        ani = animation.FuncAnimation(fig, DataThread.run(), interval=1000)
        ax1.plot(timex, eeg1, color="r")
        plt.show()"""
        timex = timedf.iloc[:,0].values  # list to store timestamps

        eeg1 = eegdf.iloc[:, 0].values  # list to store eeg data in seperate channels
        eeg2 = eegdf.iloc[:, 1].values
        eeg3 = eegdf.iloc[:, 2].values
        eeg4 = eegdf.iloc[:, 3].values
        
        plt.cla()

        plt.plot(timex, eeg1, label = "Channel 1")
        plt.plot(timex, eeg2, label = "Channel 2")
        plt.plot(timex, eeg3, label = "Channel 3")
        plt.plot(timex, eeg4, label = "Channel 4")

        plt.tight_layout()


    
    print(eeg1)

    #timex = timedf.iloc[:,0]  # list to store timestamps

    #eeg1 = eegdf.iloc[:, 0]  # list to store eeg data in seperate channels
    #eeg2 = eegdf.iloc[:, 1]
    #eeg3 = eegdf.iloc[:, 2]
    #eeg4 = eegdf.iloc[:, 3]



    board.stop_stream()
    board.release_session()


    plt.tight_layout()




    #main()
    ani = FuncAnimation(plt.gcf(), main, interval=1000)
    #plt.tight_layout()
    plt.show()
