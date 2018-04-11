# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) University of Waikato, Hamilton, NZ. All rights reserved.

# Licensed under the MIT license. See LICENSE file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import argparse
import yaml
import numpy as np
import cntk as C
from PIL import Image
import cntk
from cntk import load_model, placeholder, Constant
from cntk import Trainer, UnitType
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk.layers import Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.ops import combine, softmax
from cntk.ops.functions import CloneMethod
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import log_number_of_parameters, ProgressPrinter
from datetime import datetime
from time import sleep

base_folder = os.path.dirname(os.path.abspath(__file__))

# model parameters
base_model_file = None
feature_node_name = None
last_hidden_node_name = None
image_height = None
image_width = None
num_channels = None
features_stream_name = None
label_stream_name = None
new_output_node_name = None

# Learning parameters
max_epochs = None
mb_size = None
lr_per_mb = None
momentum_per_mb = None
l2_reg_weight = None

# input files
train_image_folder = None
test_image_folder = None
file_endings = None

# training output
results_file = None
new_model_file = None
class_map_file = None

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
    global features_stream_name
    global label_stream_name
    global new_output_node_name
    global max_epochs
    global mb_size
    global lr_per_mb
    global momentum_per_mb
    global l2_reg_weight
    global train_image_folder
    global test_image_folder
    global file_endings
    global results_file
    global new_model_file
    global class_map_file
    global prediction_in
    global prediction_out
    global prediction

    base_model_file = os.path.join(base_folder, "..", "PretrainedModels", "ResNet_18.model")
    # model setup
    feature_node_name = "features"
    last_hidden_node_name = "z.x"
    image_height = 224
    image_width = 224
    num_channels = 3
    features_stream_name = 'features'
    label_stream_name = 'labels'
    new_output_node_name = "prediction"

    # learning parameters
    max_epochs = 20
    mb_size = 50
    lr_per_mb = [0.2]*10 + [0.1]
    momentum_per_mb = 0.9
    l2_reg_weight = 0.0005

    # input files
    train_image_folder = os.path.join(base_folder, "..", "DataSets", "Animals", "Train")
    test_image_folder = os.path.join(base_folder, "..", "DataSets", "Animals", "Test")
    file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    # training output
    results_file = os.path.join(base_folder, "Output", "predictions.txt")
    new_model_file = os.path.join(base_folder, "Output", "TransferLearning.model")
    class_map_file = None

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
    global features_stream_name
    global label_stream_name
    global new_output_node_name
    global max_epochs
    global mb_size
    global lr_per_mb
    global momentum_per_mb
    global l2_reg_weight
    global train_image_folder
    global test_image_folder
    global file_endings
    global results_file
    global new_model_file
    global class_map_file

    config = yaml.load(open(cfg))
    if 'base_model_file' in config:
        base_model_file = config['base_model_file']
    if 'feature_node_name' in config:
        feature_node_name = config['feature_node_name']
    if 'last_hidden_node_name' in config:
        last_hidden_node_name = config['last_hidden_node_name']
    if 'image_height' in config:
        image_height = int(config['image_height'])
    if 'image_width' in config:
        image_width = int(config['image_width'])
    if 'num_channels' in config:
        num_channels = int(config['num_channels'])
    if 'features_stream_name' in config:
        features_stream_name = config['features_stream_name']
    if 'label_stream_name' in config:
        label_stream_name = config['label_stream_name']
    if 'new_output_node_name' in config:
        new_output_node_name = config['new_output_node_name']
    if 'max_epochs' in config:
        max_epochs = int(config['max_epochs'])
    if 'mb_size' in config:
        mb_size = int(config['mb_size'])
    if 'lr_per_mb' in config:
        lr_per_mb = list(config['lr_per_mb'])
    if 'momentum_per_mb' in config:
        momentum_per_mb = float(config['momentum_per_mb'])
    if 'l2_reg_weight' in config:
        l2_reg_weight = float(config['l2_reg_weight'])
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
    if 'class_map_file' in config:
        class_map_file = config['class_map_file']


def init_vars(args):
    """
    Initializes the variables.

    :param args: the arguments to parse
    :type args: list
    """
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


# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_width, image_height, num_channels, num_classes, randomize=True):
    transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
            features =StreamDef(field='image', transforms=transforms),
            labels   =StreamDef(field='label', shape=num_classes))),
            randomize=randomize)


# Creates the network model for transfer learning
def create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, input_features, freeze=False):
    # Load the pretrained classification net and find nodes
    base_model   = load_model(base_model_file)
    feature_node = find_by_name(base_model, feature_node_name)
    last_node    = find_by_name(base_model, last_hidden_node_name)

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Add new dense layer for class prediction
    feat_norm  = input_features - Constant(114)
    cloned_out = cloned_layers(feat_norm)
    z          = Dense(num_classes, activation=None, name=new_output_node_name) (cloned_out)

    return z


# Trains a transfer learning model
def train_model(base_model_file, feature_node_name, last_hidden_node_name,
                image_width, image_height, num_channels, num_classes, train_map_file,
                num_epochs, max_images=-1, freeze=False):
    epoch_size = sum(1 for line in open(train_map_file))
    if max_images > 0:
        epoch_size = min(epoch_size, max_images)

    # Create the minibatch source and input variables
    minibatch_source = create_mb_source(train_map_file, image_width, image_height, num_channels, num_classes)
    image_input = C.input_variable((num_channels, image_height, image_width))
    label_input = C.input_variable(num_classes)

    # Define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[features_stream_name],
        label_input: minibatch_source[label_stream_name]
    }

    # Instantiate the transfer learning model and loss function
    tl_model = create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, image_input, freeze)
    ce = cross_entropy_with_softmax(tl_model, label_input)
    pe = classification_error(tl_model, label_input)

    # Instantiate the trainer object
    lr_schedule = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch)
    mm_schedule = momentum_schedule(momentum_per_mb)
    learner = momentum_sgd(tl_model.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=num_epochs)
    trainer = Trainer(tl_model, (ce, pe), learner, progress_printer)

    # Get minibatches of images and perform model training
    print("Training transfer learning model for {0} epochs (epoch_size = {1}).".format(num_epochs, epoch_size))
    log_number_of_parameters(tl_model)
    for epoch in range(num_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            if sample_count % (100 * mb_size) == 0:
                print ("Processed {0} samples".format(sample_count))

        trainer.summarize_training_progress()

    return tl_model


# Evaluates a single image using the provided model
def eval_single_image(loaded_model, image_path, image_width, image_height):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    img = Image.open(image_path)
    if image_path.endswith("png"):
        temp = Image.new("RGB", img.size, (255, 255, 255))
        temp.paste(img, img)
        img = temp
    resized = img.resize((image_width, image_height), Image.ANTIALIAS)
    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    ## Alternatively: if you want to use opencv-python
    # cv_img = cv2.imread(image_path)
    # resized = cv2.resize(cv_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    # bgr_image = np.asarray(resized, dtype=np.float32)
    # hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    # compute model output
    arguments = {loaded_model.arguments[0]: [hwc_format]}
    output = loaded_model.eval(arguments)

    # return softmax probabilities
    sm = softmax(output[0])
    return sm.eval()


# Evaluates an image set using the provided model
def eval_test_images(loaded_model, output_file, test_map_file, image_width, image_height, max_images=-1, column_offset=0):
    num_images = sum(1 for line in open(test_map_file))
    if max_images > 0:
        num_images = min(num_images, max_images)
    print("Evaluating model output node '{0}' for {1} images.".format(new_output_node_name, num_images))

    pred_count = 0
    correct_count = 0
    np.seterr(over='raise')
    with open(output_file, 'wb') as results_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0 + column_offset]
                probs = eval_single_image(loaded_model, img_file, image_width, image_height)

                pred_count += 1
                true_label = int(tokens[1 + column_offset])
                predicted_label = np.argmax(probs)
                if predicted_label == true_label:
                    correct_count += 1

                np.savetxt(results_file, probs[np.newaxis], fmt="%.3f")
                if pred_count % 100 == 0:
                    print("Processed {0} samples ({1} correct)".format(pred_count, (float(correct_count) / pred_count)))
                if pred_count >= num_images:
                    break

    print ("{0} out of {1} predictions were correct {2}.".format(correct_count, pred_count, (float(correct_count) / pred_count)))


def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    lines = []
    for class_id in range(0, len(class_mapping)):
        folder = os.path.join(root_folder, class_mapping[class_id])
        if os.path.exists(folder):
            for entry in os.listdir(folder):
                filename = os.path.join(folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    lines.append("{0}\t{1}\n".format(filename, class_id))

    if include_unknown:
        for entry in os.listdir(root_folder):
            filename = os.path.join(root_folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                lines.append("{0}\t-1\n".format(filename))

    lines.sort()
    with open(map_file_name , 'w') as map_file:
        for line in lines:
            map_file.write(line)

    return map_file_name


def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    classes.sort()
    return np.asarray(classes)


def format_output_line(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs, class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '[{"class": "%s", "predictions": {' % true_class_name
    for i in range(0, top_n):
        line = '%s"%s":%.3f, ' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s}, "image": "%s"}]\n' % (line[:-2], img_name.replace('\\', '/').rsplit('/', 1)[1])
    return line


def train_and_eval(_base_model_file, _train_image_folder, _test_image_folder, _results_file, _new_model_file, testing = False):
    # check for model and data existence
    if not (os.path.exists(_base_model_file) and os.path.exists(_train_image_folder) and os.path.exists(_test_image_folder)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # get class mapping and map files from train and test image folder
    class_mapping = create_class_mapping_from_folder(_train_image_folder)
    np.savetxt(class_map_file, class_mapping, fmt="%s")
    train_map_file = create_map_file_from_folder(_train_image_folder, class_mapping)
    test_map_file = create_map_file_from_folder(_test_image_folder, class_mapping, include_unknown=True)

    # train
    trained_model = train_model(_base_model_file, feature_node_name, last_hidden_node_name,
                                image_width, image_height, num_channels,
                                len(class_mapping), train_map_file, num_epochs=30, freeze=True)

    if not testing:
        trained_model.save(_new_model_file)
        print("Stored trained model at %s" % _new_model_file)

    # evaluate test images
    with open(_results_file, 'w') as output_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0]
                true_label = int(tokens[1])
                probs = eval_single_image(trained_model, img_file, image_width, image_height)

                formatted_line = format_output_line(img_file, true_label, probs, class_mapping)
                output_file.write(formatted_line)

    print("Done. Wrote output to %s" % _results_file)


def predict(new_model_file, prediction_in, prediction_out):
    """
    Use the trained model to make predictions on images that appear in the "prediction_in"
    directory. Moves the incoming image into the "prediction_out" directory and places
    a JSON file (same name, only different extension) with the prediction alongside.

    :param new_model_file: the model to load
    :type new_model_file: str
    :param prediction_in: the directory for incoming images
    :type prediction_in: str
    :param prediction_out: the directory the processed images and the prediction files
    :type prediction_out: str
    """

    # load model and class mapping
    print("Loading model...")
    trained_model = C.load_model(new_model_file)
    print("Loading class mapping...")
    class_mapping = np.genfromtxt(class_map_file, dtype="str")

    print("Entering prediction mode...")
    while True:
        # only supported extensions
        files = [(prediction_in + os.sep + x) for x in os.listdir(prediction_in) if  os.path.splitext(x)[1] in file_endings]

        # no files present?
        if len(files) == 0:
            sleep(1)
            continue

        for f in files:
            start = datetime.now()
            print(start, "-", f)

            img_path = prediction_out + os.sep + os.path.basename(f)
            pred_path = prediction_out + os.sep + os.path.splitext(os.path.basename(f))[0] + ".json"

            probs = eval_single_image(trained_model, img_path, image_width, image_height)
            formatted_line = format_output_line(img_path, "?", probs, class_mapping)
            with open(pred_path, 'w') as pred_file:
                pred_file.write(formatted_line)

            timediff = datetime.now() - start
            print("  time:", timediff)


if __name__ == '__main__':
    reset_vars()
    init_vars(sys.argv[1:])
    if prediction:
        predict(new_model_file, prediction_in, prediction_out)
    else:
        train_and_eval(base_model_file, train_image_folder, test_image_folder, results_file, new_model_file)
