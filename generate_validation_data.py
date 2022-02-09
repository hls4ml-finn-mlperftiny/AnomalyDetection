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
        default_mels = 128 #Default mel_bins determined by MLCommons, FIXED/DO NOT CHANGE
        n_mels = 32 #TODO change to be from config file
        inputs = 128 #TODO change to be from config file
        frames = int(inputs/n_mels)
        skip = int(default_mels/n_mels)

        assert(inputs % n_mels == 0)
        assert(default_mels % n_mels == 0)

        test_files, y_true = com.test_file_list_generator(target_dir, id_str, dir_name='validation', ext='bin')

        print("\n============== CREATING TEST DATA FOR A MACHINE ID ==============")
        for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
            file_inputs = []
            binaryfile = np.fromfile(file_path, dtype=np.float32)
            for range_min in range(0, len(binaryfile), 128):
                single_sample = []
                for slice in range(range_min, range_min+frames*128,128):
                    single_sample.extend(binaryfile[slice:slice+default_mels:skip])
                if len(single_sample) ==inputs:
                    file_inputs.append(single_sample)
            X_machine_data.append(file_inputs)
        X.append(X_machine_data)
        y.append(y_true)
    print(len(X))
    print(len(X[0]))
    print(len(X[0][0]))

    #save validation_data
    if not os.path.exists('validation_data/anomaly_detection/'):
        os.makedirs('validation_data/anomaly_detection/')
    # np.save(convert['x_npy_plot_roc'],X)
    # np.save(convert['y_npy_plot_roc'],y)
    # np.save(convert['x_npy_hls_test_bench'],X[0][0][0:10])
    # np.save(convert['y_npy_hls_test_bench'],y[0][0:10])
    np.save(f'validation_data/anomaly_detection/{inputs}input_validation_data.npy',X)
    np.save(f'validation_data/anomaly_detection/{inputs}input_validation_data_ground_truths.npy',y)
    # np.save(convert['x_npy_hls_test_bench'],X[0][0][0:10])
    # np.save(convert['y_npy_hls_test_bench'],y[0][0:10])
                    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)