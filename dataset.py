from torch.utils.data import Dataset as TorchDataset
import numpy as np
import h5py
import av
import io

def decode_mp3(mp3_arr):
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform


def pad_or_truncate(x, audio_length):
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]

class openmic18_student(TorchDataset):
    def __init__(self, hdf5_file, length=-1, sample_rate=32000, classes_num=40, clip_length=10):
        self.hdf5_file = hdf5_file
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.classes_num = classes_num

        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['audio_name'])
            if (-1 < length) and (length < self.length):
                self.length = length

        self.clip_length = clip_length * sample_rate
        self.dataset_file = None

    def open_hdf5(self):
        self.dataset_file = h5py.File(self.hdf5_file, 'r')

    def __getitem__(self, index):
        if self.dataset_file is None:
            self.open_hdf5()

        waveform = decode_mp3(self.dataset_file['mp3'][index])
        waveform = pad_or_truncate(waveform, self.clip_length)

        target = self.dataset_file['target'][index]
        output = self.dataset_file['output'][index]

        return waveform.reshape(1, -1), target, output

    def __len__(self):
        return self.length

class openmic18(TorchDataset):
    def __init__(self, hdf5_file, length=-1, sample_rate=32000, classes_num=40, clip_length=10):
        self.hdf5_file = hdf5_file
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.classes_num = classes_num

        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['audio_name'])
            if (-1 < length) and (length < self.length):
                self.length = length

        self.clip_length = clip_length * sample_rate
        self.dataset_file = None

    def open_hdf5(self):
        self.dataset_file = h5py.File(self.hdf5_file, 'r')

    def __getitem__(self, index):
        if self.dataset_file is None:
            self.open_hdf5()

        waveform = decode_mp3(self.dataset_file['mp3'][index])
        waveform = pad_or_truncate(waveform, self.clip_length)

        target = self.dataset_file['target'][index]

        return waveform.reshape(1, -1), target

    def __len__(self):
        return self.length

def get_dataset_test(test_hdf5, length = -1, sample_rate=32000, classes_num=40, clip_length=10):
    dataset = openmic18(test_hdf5, length=length, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length)
    return dataset

def get_dataset_train(train_hdf5, length = -1, sample_rate=32000, classes_num=40, clip_length=10):
    dataset = openmic18(train_hdf5, length=length, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length)
    return dataset

def get_dataset_train_student(train_student_hdf5, length = -1, sample_rate=32000, classes_num=40, clip_length=10):
    dataset = openmic18(train_student_hdf5, length=length, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length)
    return dataset
