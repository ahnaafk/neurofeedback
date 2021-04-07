import time
import brainflow
import numpy as np
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        style.use('fivethirtyeight')
        timex = []  # list to store timestamps
        eeg1 = []  # list to store eeg data in seperate channels
        eeg2 = []
        eeg3 = []
        eeg4 = []
        while self.keep_alive:
            # get board data removes data from the buffer
            while self.board.get_board_data_count() < 250:
                time.sleep(0.005)
            data = self.board.get_board_data()
            timedf = pd.DataFrame(np.transpose(data[self.timestamp]))
            eegdf = pd.DataFrame(np.transpose(data[self.eeg_channels]))
            #print("EEG 1")
            #print(eeg1)
            #print("EEG Dataframe")
            #print(eegdf)
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
            eeg1.append(eegdf.iloc[:, 0])
            eeg2.append(eegdf.iloc[:, 1])
            eeg3.append(eegdf.iloc[:, 2])
            eeg4.append(eegdf.iloc[:, 3])
            timex.append(timedf.iloc[:, 0])

        return eeg1
        return eeg2
        return eeg3
        return eeg4
        return timex

        
def main():
    BoardShim.enable_dev_board_logger()
    # use synthetic board for demo
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.title("tay k very hard, blocboy very hard", fontsize=15)
    plt.ylabel("dick too big its like a foot is in yo mouf", fontsize=15)
    plt.xlabel(
        "\ni aint sayin she a gold digger, but shes a ho - kanye weest", fontsize=10)
    fig.show()
    #eeg1 = [4]
    #timex = [6]

    data_thread = DataThread(board, board_id)
    data_thread.start()
    try:
        time.sleep(60)
    finally:
        data_thread.keep_alive = False
        data_thread.join()
        print(eeg1)
        print(timex)
        #plt.autoscale(enable=True, axis="y", tight=True)
        ax1.plot(timex, eeg1, color="r")
        ani = animation.FuncAnimation(fig, DataThread(board, board_id).run(), interval=1000)
        plt.show()
    board.stop_stream()
    board.release_session()
if __name__ == "__main__":
    main()