from rl_agent.reader.BaseReader import BaseDataReader
import numpy as np
from rl_agent.utils import padding_and_clip
class MDPDataReader(BaseDataReader):

    def __init__(self, params):
        '''
        - from BaseReader:
            - phase
            - data: will add Position column
        '''
        super().__init__(params)
        self.max_seq_len = params['max_seq_len']

    def _read_data(self, params):
        # read data_file
        super()._read_data(params)
        print("Load item meta data")
        self.item_meta = params['item_meta']
        # self.user_info = params['user_info']
        self.item_vec_size = len(self.item_meta[0])
        # self.user_vec_size = len(self.user_info[0])

        self.item_ids = params['item_ids']

        self.id2idx = {sid: i for i, sid in enumerate(self.item_ids)}

    ###########################
    #        Helper           #
    ###########################
    def _map(self, sid):
        sid = str(sid)
        return self.id2idx[sid]

    ###########################
    #        Iterator         #
    ###########################

    def __getitem__(self, idx):
        user_ID, slate_of_items, user_feedback, user_history, sequence_id = self.data[self.phase].iloc[idx]

        exposure_raw = eval(slate_of_items)
        history_raw = eval(user_history)

        exposure = [self._map(x) for x in exposure_raw]
        history = [self._map(x) for x in history_raw]

        hist_length = len(history)
        history = padding_and_clip(history, self.max_seq_len)
        # print(f"history{}")
        feedback = eval(user_feedback)

        record = {
            'timestamp': int(1),  # timestamp is irrelevant, just a hack temporal
            'exposure': np.array(exposure).astype(int),
            'exposure_features': self.get_item_list_meta(exposure).astype(float),
            'feedback': np.array(feedback).astype(float),
            'history': np.array(history).astype(int),
            'history_features': self.get_item_list_meta(history).astype(float),
            'history_length': int(min(hist_length, self.max_seq_len)),
        }
        return record

    def get_item_list_meta(self, item_list):
        return np.array([self.item_meta[item] for item in item_list])

    def get_statistics(self):
        '''
        - n_user
        - n_item
        - s_parsity
        - from BaseReader:
            - length
            - fields
        '''
        stats = super().get_statistics()
        stats['length'] = len(self.data[self.phase])
        stats['n_item'] = len(self.item_meta)
        stats['item_vec_size'] = self.item_vec_size
        # stats['user_portrait_len'] = self.user_vec_size
        stats['max_seq_len'] = self.max_seq_len
        return stats
