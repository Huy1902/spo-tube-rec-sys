from torch.utils.data import Dataset

class BaseDataReader(Dataset):
  def __init__(self, params):
    self.phase = 'train'
    self.n_worker = params['n_worker']
    self._read_data(params)

  def _read_data(self, params):
    self.data = dict()
    self.data['train'] = params['train']
    self.data['val'] = params['val']


  def __getitem__(self, idx):
    pass

  def __len__(self):
    return len(self.data[self.phase])

  def get_statistics(self):
    return {'length': len(self)}

  def set_phase(self, phase):
    assert phase in ['train', 'val', 'test']
    self.phase = phase
