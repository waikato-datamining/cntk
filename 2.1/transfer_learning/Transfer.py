# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) University of Waikato, Hamilton, NZ. All rights reserved.

# Licensed under the MIT license. See LICENSE file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import argparse
import cntk
import yaml

base_folder = os.path.dirname(os.path.abspath(__file__))

# model parameters
base_model_file = None
feature_node_name = None
last_hidden_node_name = None
image_height = None
image_width = None
num_channels = None

# input files
train_image_folder = None
test_image_folder = None
file_endings = None

# model
results_file = None
new_model_file = None

# prediction mode parameters
prediction_in = None
prediction_out = None
prediction = False


def reset_vars():
    """
    Resets the variables to default values.
    """
    global base_model_file
    global feature_node_name
    global last_hidden_node_name
    global image_height
    global image_width
    global num_channels
    global train_image_folder
    global test_image_folder
    global file_endings
    global results_file
    global new_model_file
    global prediction_in
    global prediction_out
    global prediction

    base_model_file = os.path.join(base_folder, "..", "PretrainedModels", "ResNet_18.model")
    feature_node_name = "features"
    last_hidden_node_name = "z.x"
    image_height = 224
    image_width = 224
    num_channels = 3

    # input files
    train_image_folder = os.path.join(base_folder, "..", "DataSets", "Animals", "Train")
    test_image_folder = os.path.join(base_folder, "..", "DataSets", "Animals", "Test")
    file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    # model
    results_file = os.path.join(base_folder, "Output", "predictions.txt")
    new_model_file = os.path.join(base_folder, "Output", "TransferLearning.model")

    # prediction mode
    prediction_in = None
    prediction_out = None
    prediction = False


def cfg_from_file(cfg):
    """
    Loads the configuration from the specified file.

    :param cfg: the config file to load (yaml)
    :type cfg: str
    """
    global base_model_file
    global feature_node_name
    global last_hidden_node_name
    global image_height
    global image_width
    global num_channels
    global train_image_folder
    global test_image_folder
    global file_endings
    global results_file
    global new_model_file
    global prediction_in
    global prediction_out

    config = yaml.load(open(cfg))
    if 'base_model_file' in config:
        base_model_file = config['base_model_file']
    if 'feature_node_name' in config:
        feature_node_name = config['feature_node_name']
    if 'last_hidden_node_name' in config:
        last_hidden_node_name = config['last_hidden_node_name']
    if 'image_height' in config:
        image_height = config['image_height']
    if 'image_width' in config:
        image_width = config['image_width']
    if 'num_channels' in config:
        num_channels = config['num_channels']
    if 'train_image_folder' in config:
        train_image_folder = config['train_image_folder']
    if 'test_image_folder' in config:
        test_image_folder = config['test_image_folder']
    if 'file_endings' in config:
        file_endings = config['file_endings']
    if 'results_file' in config:
        results_file = config['results_file']
    if 'new_model_file' in config:
        new_model_file = config['new_model_file']


def init_vars(args):
    """
    Initializes the variables.

    :param args: the arguments to parse
    :type args: list
    """
    global base_model_file
    global feature_node_name
    global last_hidden_node_name
    global image_height
    global image_width
    global num_channels
    global train_image_folder
    global test_image_folder
    global file_endings
    global results_file
    global new_model_file
    global prediction_in
    global prediction_out
    global prediction

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configuration file in YAML format',
                        required=False, default=None)
    parser.add_argument('-t', '--device_type', type=str, help="The type of the device (cpu|gpu)",
                        required=False, default="cpu")
    parser.add_argument('-d', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)
    parser.add_argument('-l', '--list_devices', action='store_true', help="Lists the available devices and exits",
                        required=False, default=False)
    parser.add_argument('--prediction', action='store_true', help="Switches to prediction mode",
                        required=False, default=False)
    parser.add_argument('--prediction_in', type=str, help="The input directory for images in prediction mode",
                        required=False, default="")
    parser.add_argument('--prediction_out', type=str, help="The output directory for processed images and predicitons in prediction mode",
                        required=False, default="")

    args = vars(parser.parse_args(args=args))

    # prediction mode?
    prediction = args['prediction']
    if prediction:
        prediction_in = args['prediction_in']
        if not os.path.exists(prediction_in):
            raise RuntimeError("Prediction input directory '%s' does not exist" % prediction_in)
        prediction_out = args['prediction_out']
        if not os.path.exists(prediction_out):
            raise RuntimeError("Prediction output directory '%s' does not exist" % prediction_out)
        if prediction_in == prediction_out:
            raise RuntimeError("Input and output directories for prediction are the same: %s" % prediction_in)

    if args['list_devices']:
        print("Available devices (Type - ID - description)")
        for d in cntk.device.all_devices():
            if d.type() == 0:
                type = "cpu"
            elif d.type() == 1:
                type = "gpu"
            else:
                type = "<unknown:" + str(d.type()) + ">"
            print(type + " - " + str(d.id()) + " - " + str(d))
        sys.exit(0)
    if args['config'] is not None:
        cfg_from_file(args['config'])
    if args['device'] is not None:
        if args['device_type'] == 'gpu':
            cntk.device.try_set_default_device(cntk.device.gpu(args['device']))
        else:
            cntk.device.try_set_default_device(cntk.device.cpu())


if __name__ == '__main__':
    reset_vars()
    init_vars(sys.argv)
