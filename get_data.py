import argparse
import time

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

def main ():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
#    params.serial_port = 'COM3'
    board_id = -1
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    data = board.get_board_data()
    eeg_channel = BoardShim.get_eeg_channels(board_id)
    timestamp = BoardShim.get_timestamp_channel(board_id)
    time.sleep(5)

    board.stop_stream()
    board.release_session()

    #print (data)
    #print (eeg_channel)
    #print(timestamp)
    print(data[timestamp])


if __name__ == "__main__":
    main ()