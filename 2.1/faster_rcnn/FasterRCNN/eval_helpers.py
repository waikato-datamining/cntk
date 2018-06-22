# Copyright (c) Microsoft. All rights reserved.
# Copyright (C) University of Waikato, Hamilton, NZ

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from builtins import str
import sys
import os
import numpy as np
from builtins import range
import copy, textwrap
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
from matplotlib.pyplot import imsave
from cntk import input_variable, Axis
from utils.nms.nms_wrapper import apply_nms_to_single_image_results
from cntk_helpers import regress_rois
import cv2
from datetime import datetime
from time import sleep

available_font = "arial.ttf"
try:
    dummy = ImageFont.truetype(available_font, 16)
except:
    available_font = "FreeMono.ttf"


def write_padded_image(img_path, out_path):
    """
    Writes the padding image.

    :param img_path: the image path
    :param out_path: the output path
    """
    # read and resize image
    img_width, img_height = im_width_height(img_path)
    scale = 800.0 / max(img_width, img_height)
    img_height = int(img_height * scale)
    img_width = int(img_width * scale)
    if img_width > img_height:
        h_border = 0
        v_border = int((img_width - img_height) / 2)
    else:
        h_border = int((img_height - img_width) / 2)
        v_border = 0

    pad_color = [103, 116, 123]  # [114, 114, 114]
    cv_img = cv2.imread(img_path)
    resized = cv2.resize(cv_img, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    img_debug = cv2.copyMakeBorder(resized, v_border, v_border, h_border, h_border, cv2.BORDER_CONSTANT, value=pad_color)

    # save image
    cv2.imwrite(out_path, img_debug)


def faster_rcnn_visualize_results(img_path, roi_labels, roi_scores, roi_rel_coords, pad_width, pad_height, classes,
                                  nms_keep_indices=None, draw_negative_rois=True, decision_threshold=0.0):
    """
    Visualize results.
    :param img_path: the image path
    :param roi_labels: the ROI labels
    :param roi_scores: the scores
    :param roi_rel_coords: the relative coordinates
    :param pad_width: the padding width
    :param pad_height: the padding height
    :param classes: the classes
    :param nms_keep_indices: the indices to keep
    :param draw_negative_rois: whether to draw negative ROIs (eg background)
    :param decision_threshold: the threshold
    """

    # read and resize image
    img_width, img_height = im_width_height(img_path)
    scale = 800.0 / max(img_width, img_height)
    img_height = int(img_height * scale)
    img_width = int(img_width * scale)
    if img_width > img_height:
        h_border = 0
        v_border = int((img_width - img_height) / 2)
    else:
        h_border = int((img_height - img_width) / 2)
        v_border = 0

    pad_color = [103, 116, 123]  # [114, 114, 114]
    cv_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    img_debug = cv2.copyMakeBorder(resized, v_border, v_border, h_border, h_border, cv2.BORDER_CONSTANT, value=pad_color)
    rect_scale = 800 / pad_width

    assert(len(roi_labels) == len(roi_rel_coords))
    if roi_scores:
        assert(len(roi_labels) == len(roi_scores))
        min_score = min(roi_scores)
        print("roiScores min: {}, max: {}, threshold: {}".format(min_score, max(roi_scores), decision_threshold))
        if min_score > decision_threshold:
            decision_threshold = min_score * 0.5
            print("reset decision threshold to: {}".format(decision_threshold))

    # draw multiple times to avoid occlusions
    for it in range(0, 3):
        for roiIndex in range(len(roi_rel_coords)):
            label = roi_labels[roiIndex]
            if roi_scores:
                score = roi_scores[roiIndex]
                if decision_threshold and score < decision_threshold:
                    label = 0

            # init drawing parameters
            thickness = 1
            if label == 0:
                color = (255, 0, 0)
            else:
                color = colors_palette()[label]

            rect = [(rect_scale * i) for i in roi_rel_coords[roiIndex]]
            rect[0] = int(max(0, min(pad_width, rect[0])))
            rect[1] = int(max(0, min(pad_height, rect[1])))
            rect[2] = int(max(0, min(pad_width, rect[2])))
            rect[3] = int(max(0, min(pad_height, rect[3])))

            # draw in higher iterations only the detections
            if (it == 0) and draw_negative_rois:
                draw_rectangles(img_debug, [rect], color=color, thickness=thickness)
            elif (it == 1) and (label > 0):
                if not nms_keep_indices or (roiIndex in nms_keep_indices):
                    thickness = 4
                draw_rectangles(img_debug, [rect], color=color, thickness=thickness)
            elif (it == 2) and (label > 0):
                if not nms_keep_indices or (roiIndex in nms_keep_indices):
                    font = ImageFont.truetype(available_font, 18)
                    text = classes[label]
                    if roi_scores:
                        text += "(" + str(round(score, 2)) + ")"
                    img_debug = draw_text(img_debug, (rect[0], rect[1]), text, color = (255, 255, 255), font = font, color_background=color)
    return img_debug


def load_resize_and_pad(image_path, width, height, pad_value=114, image_type="default"):
    """
    Loads and resizes the image.

    :param image_path: the image to load
    :param width: the target width
    :param height: the target height
    :param pad_value: the padding value (grayscale)
    :param image_type: the image type: default, center, 2x2
    :return: (resized_with_pad, array, dims)
    """

    if "@" in image_path:
        print("WARNING: zipped image archives are not supported for visualizing results.")
        exit(0)

    img = cv2.imread(image_path)
    img_width = len(img[0])
    img_height = len(img)
    scale_w = img_width > img_height
    target_w = width
    target_h = height

    if scale_w:
        target_h = int(np.round(img_height * float(width) / float(img_width)))
    else:
        target_w = int(np.round(img_width * float(height) / float(img_height)))

    resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)

    top = int(max(0, np.round((height - target_h) / 2)))
    left = int(max(0, np.round((width - target_w) / 2)))
    bottom = height - top - target_h
    right = width - left - target_w
    resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])

    # transpose(2,0,1) converts the image to the HWC format which CNTK accepts
    model_arg_rep = np.ascontiguousarray(np.array(resized_with_pad, dtype=np.float32).transpose(2, 0, 1))

    dims = (width, height, target_w, target_h, img_width, img_height)
    return resized_with_pad, model_arg_rep, dims


def save_rois_to_file(regressed_rois, nms_keep_indices, labels, str_labels, scores, results_base_path, img_path,
                      headers=True, output_width_height=False, suppressed_labels=(), dims=None):
    """
    Stores the ROIs in a CSV file.

    :param regressed_rois: the ROIs to store
    :param nms_keep_indices: the indices to keep
    :param labels: the labels
    :param str_labels: the string labels
    :param scores: the scores
    :param results_base_path: the base output directory
    :param img_path: the image file
    :param headers: whether to output the headers
    :param output_width_height: whether to output x/y/width/height or x0/y0/x1/y1
    :param suppressed_labels: the labels to keep from being output
    :param dims: the image dimensions, as returned by "load_resize_and_pad":
                 cntk_width, cntk_height, actual_cntk_width, actual_cntk_height, original_width, original_height
    :return:
    """
    roi_path = "{}/{}-rois.csv".format(results_base_path, os.path.splitext(os.path.basename(img_path))[0])
    with open(roi_path, "w") as roi_file:
        # headers?
        if headers:
            if output_width_height:
                roi_file.write("file,x,y,w,h,label,label_str,score\n")
            else:
                roi_file.write("file,x0,y0,x1,y1,label,label_str,score\n")
        # rois
        for index in nms_keep_indices:
            if str_labels[labels[index]] in suppressed_labels:
                continue

            # get coordinates
            x0 = regressed_rois[index][0]
            y0 = regressed_rois[index][1]
            x1 = regressed_rois[index][2]
            y1 = regressed_rois[index][3]
            w = x1 - x0 + 1
            h = y1 - y0 + 1

            # translate into realworld coordinates again
            if dims is not None:
                cntk_w, cntk_h, _, _, orig_w, orig_h = dims
                aspect = orig_w / orig_h
                cntk_act_h = round(cntk_w / aspect)
                cntk_trans_w = 0
                cntk_trans_h = -round((cntk_h - cntk_act_h) / 2)
                cntk_scale = orig_w / cntk_w
                # translate
                x0 = x0 + cntk_trans_w
                y0 = y0 + cntk_trans_h
                x1 = x1 + cntk_trans_w
                y1 = y1 + cntk_trans_h
                # scale
                x0 = x0 * cntk_scale
                y0 = y0 * cntk_scale
                x1 = x1 * cntk_scale
                y1 = y1 * cntk_scale
                w = w * cntk_scale
                h = h * cntk_scale

            # output
            if output_width_height:
                roi_file.write("{},{},{},{},{},{},{},{}\n".format(
                    os.path.basename(img_path),
                    x0, y0, w, h,
                    labels[index], str_labels[labels[index]], scores[index]))
            else:
                roi_file.write("{},{},{},{},{},{},{},{}\n".format(
                    os.path.basename(img_path),
                    x0, y0, x1, y1,
                    labels[index], str_labels[labels[index]], scores[index]))


def faster_rcnn_eval_and_plot(eval_model, num_images_to_plot, test_map_file, class_map_file, img_shape,
                              results_base_path, feature_node_name, classes,
                              draw_unregressed_rois=False, draw_negative_rois=False,
                              nms_threshold=0.5, nms_conf_threshold=0.0, bgr_plot_threshold = 0.8,
                              headers=True, output_width_height=False, suppressed_labels=()):
    """
    Tests a Faster R-CNN model and plots images with detected boxes.

    :param eval_model: the model to evaluate
    :param num_images_to_plot: the number of images to plot
    :param test_map_file: the map file with the test images/labels
    :param class_map_file: the class map file
    :param img_shape: the shape
    :param results_base_path: the base of the output directory
    :param feature_node_name: the feature node name
    :param classes:
    :param draw_unregressed_rois: whether to draw unregressed ROIs
    :param draw_negative_rois: whether to draw negative ROIs (eg background)
    :param nms_threshold:
    :param nms_conf_threshold:
    :param bgr_plot_threshold:
    :param headers: whether to output the headers in the ROI file
    :param output_width_height: whether to output x/y/width/height or x0/y0/x1/y1
    """

    # load labels
    str_labels = load_class_labels(class_map_file)
    print("Class labels: %s" % str(str_labels))

    # get image paths
    with open(test_map_file) as f:
        content = f.readlines()
    img_base_path = os.path.dirname(os.path.abspath(test_map_file))
    img_file_names = [os.path.join(img_base_path, x.split('\t')[1]) for x in content]

    # prepare model
    image_input = input_variable(img_shape, dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
    frcn_eval = eval_model(image_input, dims_input)

    # dims_input_const = cntk.constant([image_width, image_height, image_width, image_height, image_width, image_height], (1, 6))
    print("Plotting results from Faster R-CNN model for %s images." % num_images_to_plot)
    for i in range(0, num_images_to_plot):
        img_path = img_file_names[i]

        # save padded image
        write_padded_image(img_path, "{}/{}-padded.jpg".format(results_base_path, os.path.splitext(os.path.basename(img_path))[0]))

        # evaluate single image
        _, cntk_img_input, dims = load_resize_and_pad(img_path, img_shape[2], img_shape[1])

        dims_input = np.array(dims, dtype=np.float32)
        dims_input.shape = (1,) + dims_input.shape
        output = frcn_eval.eval({frcn_eval.arguments[0]: [cntk_img_input], frcn_eval.arguments[1]: dims_input})

        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1).tolist()

        if draw_unregressed_rois:
            # plot results without final regression
            img_debug = faster_rcnn_visualize_results(img_path, labels, scores, out_rpn_rois, img_shape[2], img_shape[1],
                                                      classes, nms_keep_indices=None, draw_negative_rois=draw_negative_rois,
                                                      decision_threshold=bgr_plot_threshold)
            imsave("{}/{}".format(results_base_path, os.path.basename(img_path)), img_debug)

        # apply regression and nms to bbox coordinates
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

        nms_keep_indices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                             nms_threshold=nms_threshold,
                                                             conf_threshold=nms_conf_threshold)

        save_rois_to_file(regressed_rois, nms_keep_indices, labels, str_labels, scores,
                          results_base_path, img_path, headers=headers, output_width_height=output_width_height,
                          suppressed_labels=suppressed_labels, dims=dims)

        img = faster_rcnn_visualize_results(img_path, labels, scores, regressed_rois, img_shape[2], img_shape[1],
                                            classes, nms_keep_indices=nms_keep_indices,
                                            draw_negative_rois=draw_negative_rois,
                                            decision_threshold=bgr_plot_threshold)
        imsave("{}/{}-regr.jpg".format(results_base_path, os.path.splitext(os.path.basename(img_path)))[0], img)


def faster_rcnn_eval(eval_model, test_map_file, class_map_file, img_shape,
                     results_base_path, feature_node_name,
                     nms_threshold=0.5, nms_conf_threshold=0.0,
                     headers=True, output_width_height=False, suppressed_labels=()):
    """
    Tests a Faster R-CNN model and outputs rois.

    :param eval_model: the model
    :param test_map_file: the map file with the test images/labels
    :param class_map_file: the class map file
    :param img_shape: the shape
    :param results_base_path: the base directory for the results
    :param feature_node_name: the feature node name
    :param nms_threshold:
    :param nms_conf_threshold:
    :param headers: whether to output the headers in the ROI file
    :param output_width_height: whether to output x/y/width/height or x0/y0/x1/y1
    :param suppressed_labels: the labels to suppress from being output in the ROI files
    """

    # load labels
    str_labels = load_class_labels(class_map_file)
    print("Class labels: %s" % str(str_labels))

    # get image paths
    with open(test_map_file) as f:
        content = f.readlines()
    img_base_path = os.path.dirname(os.path.abspath(test_map_file))
    img_file_names = [os.path.join(img_base_path, x.split('\t')[1]) for x in content]

    # prepare model
    image_input = input_variable(img_shape, dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
    frcn_eval = eval_model(image_input, dims_input)

    print("Outputting results from Faster R-CNN model for %s images." % len(img_file_names))
    for i in range(0, len(img_file_names)):
        start = datetime.now()
        img_path = img_file_names[i]
        print(str(i+1) + "/" + str(len(img_file_names)) + ": " + img_file_names[i])

        # evaluate single image
        _, cntk_img_input, dims = load_resize_and_pad(img_path, img_shape[2], img_shape[1])

        dims_input = np.array(dims, dtype=np.float32)
        dims_input.shape = (1,) + dims_input.shape
        output = frcn_eval.eval({frcn_eval.arguments[0]: [cntk_img_input], frcn_eval.arguments[1]: dims_input})

        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1).tolist()

        # apply regression and nms to bbox coordinates
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)
        nms_keep_indices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                             nms_threshold=nms_threshold,
                                                             conf_threshold=nms_conf_threshold)

        save_rois_to_file(regressed_rois, nms_keep_indices, labels, str_labels, scores,
                          results_base_path, img_path, headers=headers, output_width_height=output_width_height,
                          suppressed_labels=suppressed_labels, dims=dims)

        timediff = datetime.now() - start
        print("  time:", timediff)


def faster_rcnn_pred(eval_model, prediction_in, prediction_out, class_map_file, img_shape,
                     feature_node_name, nms_threshold=0.5, nms_conf_threshold=0.0, headers=True,
                     output_width_height=False, suppressed_labels=()):
    """
    Performs predictions with a model in continuous mode.

    :param eval_model: the model to use
    :param prediction_in: the input dir for images (list)
    :param prediction_out: the output dir for images and ROI CSV files (list)
    :param class_map_file: the class map file to use
    :param img_shape: the shpage
    :param feature_node_name: the name of the feature node
    :param nms_threshold:
    :param nms_conf_threshold:
    :param headers: whether to output the headers in the ROI file
    :param output_width_height: whether to output x/y/width/height or x0/y0/x1/y1
    :param suppressed_labels: the labels to suppress from being output in the ROI files
    """

    # load labels
    str_labels = load_class_labels(class_map_file)
    print("Class labels: %s" % str(str_labels))

    # prepare model
    image_input = input_variable(img_shape, dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
    frcn_eval = eval_model(image_input, dims_input)

    while True:
        any = False

        # iterate directory pairs
        for p in range(len(prediction_in)):
            pred_in = prediction_in[p]
            pred_out = prediction_out[p]

            # only png and jpg files
            files = [(pred_in + os.sep + x) for x in os.listdir(pred_in) if (x.lower().endswith(".png") or x.lower().endswith(".jpg"))]

            for f in files:
                start = datetime.now()
                print(start, "-", f)

                img_path = pred_out + os.sep + os.path.basename(f)
                roi_path = pred_out + os.sep + os.path.splitext(os.path.basename(f))[0] + ".csv"

                cntk_img_input = None
                try:
                    _, cntk_img_input, dims = load_resize_and_pad(f, img_shape[2], img_shape[1])
                except Exception as e:
                    print(str(e))

                try:
                    # delete any existing old files in output dir
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except:
                            print("Failed to remove existing image in output directory: ", img_path)
                    if os.path.exists(roi_path):
                        try:
                            os.remove(roi_path)
                        except:
                            print("Failed to remove existing ROI file in output directory: ", roi_path)
                    # move into output dir
                    os.rename(f, img_path)
                except:
                    img_path = None

                if cntk_img_input is None:
                    continue
                if img_path is None:
                    continue

                dims_input = np.array(dims, dtype=np.float32)
                dims_input.shape = (1,) + dims_input.shape
                output = frcn_eval.eval({frcn_eval.arguments[0]: [cntk_img_input], frcn_eval.arguments[1]: dims_input})

                out_dict = dict([(k.name, k) for k in output])
                out_cls_pred = output[out_dict['cls_pred']][0]
                out_rpn_rois = output[out_dict['rpn_rois']][0]
                out_bbox_regr = output[out_dict['bbox_regr']][0]

                labels = out_cls_pred.argmax(axis=1)
                scores = out_cls_pred.max(axis=1).tolist()

                # apply regression and nms to bbox coordinates
                regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)
                nms_keep_indices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                                     nms_threshold=nms_threshold,
                                                                     conf_threshold=nms_conf_threshold)

                save_rois_to_file(regressed_rois, nms_keep_indices, labels, str_labels, scores,
                                  pred_out, img_path, headers=headers, output_width_height=output_width_height,
                                  suppressed_labels=suppressed_labels, dims=dims)

                timediff = datetime.now() - start
                print("  time:", timediff)

        # nothing processed at all, lets wait for files to appear
        if not any:
            sleep(1)


def load_class_labels(class_map_file):
    """
    Loads the labels from the class map file and stores them
    in a dictionary with their label index as the key.

    :param class_map_file: the file with the class map (label<TAB>index)
    :type class_map_file: str
    :return: the label dictionary
    :rtype: dict
    """
    result = {}
    with open(class_map_file, 'r') as map:
        for line in map.readlines():
            label, num = line.split("\t")
            result[int(num)] = label
    return result


####################################
# helper library
####################################

def im_read(img_path, throw_error_if_exif_rotation_tag_set=True):
    if not os.path.exists(img_path):
        print("ERROR: image path does not exist.")
        cv2.error

    rotation = rotation_from_exif_tag(img_path)
    if throw_error_if_exif_rotation_tag_set and (rotation != 0):
        print ("Error: exif roation tag set, image needs to be rotated by %d degrees." % rotation)
    img = cv2.imread(img_path)
    if img is None:
        print ("ERROR: cannot load image " + img_path)
        cv2.error
    if rotation != 0:
        img = cv2.imrotate(img, -90).copy()  # got this error occassionally without copy "TypeError: Layout of the output array img is incompatible with cv::Mat"
    return img


def rotation_from_exif_tag(img_path):
    tags_inverted = {v: k for k, v in TAGS.items()}
    orientation_exif_id = tags_inverted['Orientation']
    try:
        image_exif_tags = Image.open(img_path)._getexif()
    except:
        image_exif_tags = None

    # rotate the image if orientation exif tag is present
    rotation = 0
    if (image_exif_tags is not None) and (orientation_exif_id is not None) and (orientation_exif_id in image_exif_tags):
        orientation = image_exif_tags[orientation_exif_id]
        # print ("orientation = " + str(imageExifTags[orientationExifId]))
        if orientation == 1 or orientation == 0:
            rotation = 0 # no need to do anything
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            print ("ERROR: orientation = " + str(orientation) + " not_supported!")
            cv2.error
    return rotation


def im_write(img, imgPath):
    cv2.imwrite(imgPath, img)


def im_resize(img, scale, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)


def im_resize_max_dim(img, max_dim, upscale=False, interpolation=cv2.INTER_LINEAR):
    scale = 1.0 * max_dim / max(img.shape[:2])
    if (scale < 1) or upscale:
        img = im_resize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale


def im_width(img):
    return im_width_height(img)[0]


def im_height(img):
    return im_width_height(img)[1]


def im_width_height(img):
    width, height = Image.open(img).size  # this does not load the full image
    return width, height


def im_array_width(img):
    return im_array_width_height(img)[0]


def im_array_height(img):
    return im_array_width_height(img)[1]


def im_array_width_height(img):
    width = img.shape[1]
    height = img.shape[0]
    return width,height


def im_show(img, wait_duration=0, max_dim=None, window_name='img'):
    if isinstance(img, str):  # test if 'img' is a string
        img = cv2.imread(img)
    if max_dim is not None:
        scale_val = 1.0 * max_dim / max(img.shape[:2])
        if scale_val < 1:
            img = im_resize(img, scale_val)
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_duration)


def draw_rectangles(img, rects, color=(0, 255, 0), thickness=2):
    for rect in rects:
        pt1 = tuple(to_integers(rect[0:2]))
        pt2 = tuple(to_integers(rect[2:]))
        try:
            cv2.rectangle(img, pt1, pt2, color, thickness)
        except:
            import pdb; pdb.set_trace()
            print("Unexpected error:", sys.exc_info()[0])


def draw_crossbar(img, pt):
    (x,y) = pt
    cv2.rectangle(img, (0, y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, 0), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (img.shape[1],y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, img.shape[0]), (x, y), (255, 255, 0), 1)


def pt_clip(pt, max_width, max_height):
    pt = list(pt)
    pt[0] = max(pt[0], 0)
    pt[1] = max(pt[1], 0)
    pt[0] = min(pt[0], max_width)
    pt[1] = min(pt[1], max_height)
    return pt


def draw_text(img, pt, text, text_width=None, color=(255, 255, 255), color_background=None,
              font=ImageFont.truetype("arial.ttf", 16)):
    pil_img = imconvert_cv_to_pil(img)
    pil_img = pil_draw_text(pil_img, pt, text, text_width, color, color_background, font)
    return imconvert_pil_to_cv(pil_img)


def pil_draw_text(pil_img, pt, text, text_width=None, color=(255, 255, 255), color_background=None,
                  font=ImageFont.truetype("arial.ttf", 16)):
    text_y = pt[1]
    draw = ImageDraw.Draw(pil_img)
    if text_width is None:
        lines = [text]
    else:
        lines = textwrap.wrap(text, width=text_width)
    for line in lines:
        width, height = font.getsize(line)
        if color_background is not None:
            draw.rectangle((pt[0], pt[1], pt[0] + width, pt[1] + height), fill=tuple(color_background[::-1]))
        draw.text(pt, line, fill=tuple(color), font=font)
        text_y += height
    return pil_img


def colors_palette():
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
    for i in range(5):
        for dim in range(0,3):
            for s in (0.25, 0.5, 0.75):
                if colors[i][dim] != 0:
                    new_color = copy.deepcopy(colors[i])
                    new_color[dim] = int(round(new_color[dim] * s))
                    colors.append(new_color)
    return colors


def imconvert_pil_to_cv(pil_img):
    rgb = pil_img.convert('RGB')
    return np.array(rgb).copy()[:, :, ::-1]


def imconvert_cv_to_pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)


def to_integers(list_1d):
    return [int(float(x)) for x in list_1d]


def get_dictionary(keys, values, convert_value_to_int=True):
    dictionary = {}
    for key, value in zip(keys, values):
        if convert_value_to_int:
            value = int(value)
        dictionary[key] = value
    return dictionary


class Bbox:
    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left, top, right, bottom):
        self.left = int(round(float(left)))
        self.top = int(round(float(top)))
        self.right = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        return "Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}".format(self.left, self.top, self.right, self.bottom)

    def __repr__(self):
        return str(self)

    def rect(self):
        return [self.left, self.top, self.right, self.bottom]

    def max(self):
        return max([self.left, self.top, self.right, self.bottom])

    def min(self):
        return min([self.left, self.top, self.right, self.bottom])

    def width(self):
        width  = self.right - self.left + 1
        assert(width >= 0)
        return width

    def height(self):
        height = self.bottom - self.top + 1
        assert(height >= 0)
        return height

    def surface_area(self):
        return self.width() * self.height()
