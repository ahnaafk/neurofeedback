import time
import brainflow
import numpy as np
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

class DataThread (threading.Thread):

    def __init__ (self, board, board_id):
        threading.Thread.__init__ (self)
        self.eeg_channels = BoardShim.get_eeg_channels (board_id)
        self.sampling_rate = BoardShim.get_sampling_rate (board_id)
        self.keep_alive = True
        self.board = board




def main ():
    BoardShim.enable_dev_board_logger ()

    # use synthetic board for demo
    params = BrainFlowInputParams ()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim (board_id, params)
    eeg_channels = board.get_eeg_channels(board_id)
    sampling_rate = 250
    board.prepare_session ()
    board.start_stream ()

    window_size = 5
    sleep_time = 1
    points_per_update = window_size * sampling_rate
    while True:
        time.sleep (sleep_time)
        # get current board data doesnt remove data from the buffer
        data = board.get_current_board_data (int (points_per_update))
        print ('Data Shape %s' % (str (data.shape)))
        for channel in eeg_channels:
            # filters work in-place
            DataFilter.perform_bandstop (data[channel], sampling_rate, 50.0, 4.0, 4,
                FilterTypes.BUTTERWORTH.value, 0) # bandstop 48-52
            DataFilter.perform_bandstop (data[channel], sampling_rate, 60.0, 4.0, 4,
                FilterTypes.BUTTERWORTH.value, 0) # bandstop 58 - 62
            DataFilter.perform_bandpass (data[channel], sampling_rate, 21.0, 20.0, 4,
                FilterTypes.BESSEL.value, 0) # bandpass 11 - 31

    board.stop_stream ()
    board.release_session ()

if __name__ == "__main__":
    main ()