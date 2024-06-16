import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import measure
from scipy.spatial import distance


def calculate_accuracy(output, label):
    if len(label.shape) != 1:
        label = label.reshape(-1)
    if len(output.shape) != 1:
        output = output.reshape(-1)
    label = label.astype(np.uint8)
    output = output.astype(np.uint8)

    matrix = confusion_matrix(label, output, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    OA = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    P = tp / (tp + fp) if (tp + fp) != 0 else 0
    R = tp / (tp + fn) if (tp + fn) != 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
    IOU = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    return OA, F1, IOU


def calculate_over_under(output, label):
    """
    output & label size: (b, h, w) or (h, w)
    """
    global over, under, total
    assert output.shape == label.shape
    if len(output.shape) == 2:
        over, under, total = over_under_classification(output, label)
    elif len(output.shape) == 3:
        batch_size = output.shape[0]
        results_over = []
        results_under = []
        results_total = []
        for b in range(batch_size):
            output_batch = output[b]
            label_batch = label[b]
            result_over, result_under, result_total = over_under_classification(output_batch, label_batch)
            results_over.append(result_over)
            results_under.append(result_under)
            results_total.append(result_total)
        over = np.mean([x for x in results_over if x is not None])
        under = np.mean([x for x in results_under if x is not None])
        total = np.mean([x for x in results_total if x is not None])
    else:
        print('Shape not satisfied. Output shape: ', output.shape)
    return over, under, total


def over_under_classification(output, label):
    """
    Calculate indicators for over segmentation and under segmentation
    :param output: 0/1 binary map of (h, w) shape
    :param label: 0/1 binary map of (h, w) shape
    :return: over segmentation, under segmentation, total
    """
    # assert len(output.shape) == 2
    # assert len(label.shape) == 2

    global_area_output = np.sum(output)

    labeled_output = measure.label(output, background=0, connectivity=1)  # region analysis
    props_output = measure.regionprops(labeled_output)  # get region attribute
    labeled_label = measure.label(label, background=0, connectivity=1)
    props_label = measure.regionprops(labeled_label)
    # print(len(props_output), len(props_label))

    if len(props_output) == 0 or len(props_label) == 0:
        # ignore None value when calculate Mean value: np.mean([x for x in my_list if x is not None])
        return None, None, None

    # save results in List
    results_over = []
    results_under = []
    results_total = []

    for prop_output in props_output:
        min_distance = float('inf')
        closest_prop_label = None

        for prop_label in props_label:
            dist = distance.euclidean(prop_output.centroid, prop_label.centroid)

            if dist < min_distance:
                min_distance = dist
                closest_prop_label = prop_label

        area_output = prop_output.area
        area_label = closest_prop_label.area

        region_output = np.where(labeled_output == prop_output.label, 1, 0)
        region_label = np.where(labeled_label == closest_prop_label.label, 1, 0)
        area_intersection = np.sum(region_output * region_label)
        # print(area_output, area_label, area_intersection)

        over_seg = 1 - (area_intersection / area_label)
        under_seg = 1 - (area_intersection / area_output)
        total_seg = np.sqrt((np.square(over_seg) + np.square(under_seg)) / 2)
        # print(over_seg, under_seg, total_seg)

        global_over_seg = over_seg * area_output / global_area_output
        global_under_seg = under_seg * area_output / global_area_output
        global_total_seg = total_seg * area_output / global_area_output
        # print(global_over_seg, global_under_seg, global_total_seg)

        results_over.append(global_over_seg)
        results_under.append(global_under_seg)
        results_total.append(global_total_seg)

    result_over = np.sum(results_over)
    result_under = np.sum(results_under)
    result_total = np.sum(results_total)
    # print(result_over, result_under, result_total)

    return result_over, result_under, result_total
