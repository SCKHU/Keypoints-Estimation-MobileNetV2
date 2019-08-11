from easydict import EasyDict as edict
import json
import argparse
import os
import pickle
import tensorflow as tf
from pprint import pprint
import sys


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="MobileNet TensorFlow Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')


def pred_pts(preds, order='bchw'):

    if order == 'bchw': #change order to bhwc
        preds = np.transpose(preds, (0,2,3,1))
    batch_size = preds.shape[0]
    output_size = preds.shape[1]
    num_class = preds.shape[3]//3
    keypoints = np.zeros((num_class, 2 )  )    
    
    heatmaps = preds[:,:,:,0:num_class]
    offsets_x = preds[:,:,:,num_class:num_class*2]
    offsets_y = preds[:,:,:,num_class*2:num_class*3]

    for b in range(batch_size):
        heatmap = heatmaps[b]
        offset_x = offsets_x[b]*8.0
        offset_y = offsets_y[b]*8.0
        
        X1 = np.linspace(1, output_size, output_size)
        [X, Y] = np.meshgrid(X1, X1)

        for k in range(num_class):
            weight = heatmap[:,:,k]
            weight[weight < 0.01] = 0
            pos_x = (X + offset_x[:,:,k])*weight
            pos_y = (Y + offset_y[:,:,k])*weight

            keypoints[k][0] = pos_x.sum()/(weight.sum() + 0.0000000001)
            keypoints[k][1] = pos_y.sum()/(weight.sum() + 0.0000000001) 

    return keypoints / output_size
