import common as com
import os
import numpy as np
import argparse
from tqdm import tqdm

def main(args):
    # load parameter.yaml
    param = com.yaml_load(args.config)
    convert = param.get("convert", None)
    param = param["train"]

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    target_dir = com.select_dirs(param=param)[0]
    print("target dir is {}".format(target_dir))
    machine_type = os.path.split(target_dir)[1]
    print(machine_type)
    machine_id_list = com.get_machine_id_list_for_test(target_dir, dir_name='validation', ext='bin')
    print(machine_id_list)

    X = []
    y = []
    downsample = param['feature']['downsample']
    input_dim = param['model']['input_dim']
    n_mels = param['feature']['n_mels']

    for id_str in machine_id_list:
        print('id_str: {}'.format(id_str))
        # load test file
        X_machine_data = []
        y_machine_data = []
        test_files, y_true = com.test_file_list_generator(target_dir, id_str, dir_name='validation', ext='bin')

        print("\n============== CREATING TEST DATA FOR A MACHINE ID ==============")
        for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
            data = np.fromfile(file_path)
            data = data.astype('float')
            print(data)
            print('len_data: {}'.format(len(data)))
            exit(-1)
        quit(-1)
    #         if downsample:
    #             new_mels = 32
    #             new_frames = int(input_dim/new_mels)
    #             increment = int(n_mels / new_mels) #value by which to sample the full 640. 
    #             offset = n_mels - new_mels*increment #ensures we sample something that is within the expected size
    #             assert(input_dim % new_mels == 0)

    #             vector_array = np.zeros((vector_array_size, new_mels*new_frames))

    #             for t in range(new_frames):
    #                 new_vec = log_mel_spectrogram[:, t: t + vector_array_size].T
    #                 vector_array[:, new_mels * t: new_mels * (t + 1)] = new_vec[:,offset::increment]

    #     return vector_array
    #         X_machine_data.append(data)
    #     X.append(X_machine_data)
    #     y.append(y_true)
    
    # #save validation_data
    # if not os.path.exists('validation_data/anomaly_detection/'):
    #     os.makedirs('validation_data/anomaly_detection/')
    # np.save(convert['x_npy_plot_roc'],X)
    # np.save(convert['y_npy_plot_roc'],y)
    # np.save(convert['x_npy_hls_test_bench'],X[0][0][0:10])
    # np.save(convert['y_npy_hls_test_bench'],y[0][0:10])

                    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)