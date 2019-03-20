# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import argparse
import cntk
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results

from FasterRCNN_config import cfg as detector_cfg
from FasterRCNN_config import cfg_from_file

def set_vars():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file in YAML format', required=True, default=None)
    args = vars(parser.parse_args())

# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    set_vars()
    yaml_config = cfg_from_file(args['config'])
    cfg = merge_configs([detector_cfg, yaml_config])
    prepare(cfg, False)
    cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))

    # train and test
    trained_model = train_faster_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)

    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # Plot results on test set images
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)

    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)


