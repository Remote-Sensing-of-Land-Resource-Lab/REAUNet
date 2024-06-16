import os
from skimage import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def listdir(folder, endswith='.tif'):
    """
    find all tiles end with '.tif'
    """
    all_files = os.listdir(folder)
    select_files = [file for file in all_files if file.endswith(endswith)]
    return select_files


def drop_edge(image, drop_pixels: int = 8):
    """
    image.shape: (h, w, c) or (h, w)
    """
    if len(image.shape) == 3:
        image = image[drop_pixels:-drop_pixels, drop_pixels:-drop_pixels, :]
    elif len(image.shape) == 2:
        image = image[drop_pixels:-drop_pixels, drop_pixels:-drop_pixels]
    return image


def crop_single_image(image,
                      label,
                      name_prefix,
                      save_image_folder,
                      save_label_folder,
                      crop_size: int,
                      crop_step: int):
    """
    crop a single image to samples
    image: (h, w, c) uint8
    label: (h, w) uint8
    """
    assert len(image.shape) == 3
    assert len(label.shape) == 2

    rows = (image.shape[0] - crop_size) // crop_step
    cols = (image.shape[1] - crop_size) // crop_step
    # print(image.shape, rows, cols)

    for i in range(rows):
        for j in range(cols):
            image_crop = image[i * crop_step:i * crop_step + crop_size,
                               j * crop_step:j * crop_step + crop_size, :]
            label_crop = label[i * crop_step:i * crop_step + crop_size,
                               j * crop_step:j * crop_step + crop_size]

            image_crop = image_crop.astype(np.uint8)
            label_crop = label_crop.astype(np.uint8)

            save_name = name_prefix + '_h%02dw%02d.tif' % (i + 1, j + 1)
            save_image_path = os.path.join(save_image_folder, save_name)
            save_label_path = os.path.join(save_label_folder, save_name)
            io.imsave(save_image_path, image_crop)
            io.imsave(save_label_path, label_crop)
            print('\r' + f'[{0:d}/{1:d}] {2}'.format(i * cols + j + 1, rows * cols, save_name), end='')


def data_crop(image_folder,
              label_folder,
              save_image_folder,
              save_label_folder,
              crop_size: int,
              crop_step: int):
    tif_files = listdir(image_folder)
    print(tif_files[0], len(tif_files))

    # make new directory
    if not os.path.exists(save_image_folder):
        os.mkdir(save_image_folder)
    if not os.path.exists(save_label_folder):
        os.mkdir(save_label_folder)

    for idx, tif_name in enumerate(tif_files):
        image_path = os.path.join(image_folder, tif_name)
        label_path = os.path.join(label_folder, tif_name)

        image = io.imread(image_path)
        label = io.imread(label_path)

        image = drop_edge(image)
        label = drop_edge(label)

        image = np.nan_to_num(image, nan=0)
        label = np.nan_to_num(label, nan=0)

        tif_name_prefix = tif_name[:-4]
        crop_single_image(image, label, tif_name_prefix,
                          save_image_folder, save_label_folder,
                          crop_size, crop_step)
        print(f'<{0:d}//{1:d}> Image Cropped: {2:s}'.format(idx + 1, len(tif_files), tif_name))
