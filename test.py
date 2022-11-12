import h5py
import regex as re

dataset_path = "/home/oscar/openmic18/data/"

def main():
    output_dict = dict()
    with open(dataset_path + 'openmic18_outputs.txt', 'r') as outputs:
        for line in outputs:
            for key, value in re.findall(r'(.*) \[(.*)\]', line):
                break
            key = bytes(key, 'utf-8')
            output_dict[key] = value
    
    with h5py.File(dataset_path + 'mp3/openmic_train.csv_mp3.hdf') as hdf_in:
        with h5py.File(dataset_path + 'mp3/openmic_train_student.csv_mp3.hdf', 'w') as hdf_out:
            audio_name_in = hdf_in['audio_name']
            mp3_in = hdf_in['mp3']
            target_in = hdf_in['target']

            audio_name_out = hdf_out.create_dataset('audio_name', shape=audio_name_in.shape, dtype=audio_name_in.dtype)
            mp3_out = hdf_out.create_dataset('mp3', shape=mp3_in.shape, dtype=mp3_in.dtype)
            target_out = hdf_out.create_dataset('target', shape=target_in.shape, dtype=target_in.dtype)

            for index, (audio_name, mp3, target)  in enumerate(zip(audio_name_in, mp3_in, target_in)):
                output = output_dict[audio_name]

                audio_name_out[index] = audio_name
                mp3_out[index] = mp3
                target_out[index] = target
                output_out[index] = output




if __name__ == '__main__':
    main()

