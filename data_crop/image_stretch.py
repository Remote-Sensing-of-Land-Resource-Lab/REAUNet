import numpy as np
from osgeo import gdal
from project_array import project_array_and_save


def image_stretch_2d(array, tmin, tmax):
    """
    array.shape: (h, w)
    """
    assert len(array.shape) == 2
    array = np.nan_to_num(array, nan=0)
    array_mask = np.where(array == 0, 0, 1).astype(np.uint8)

    array = (array.astype(np.float32) - tmin) / (tmax - tmin)
    array = np.maximum(np.minimum(array * 255, 255), 0).astype(np.uint8)
    array = array * array_mask

    return array


def image_stretch(image_path, save_path,
                  min_percentage=0.02, max_percentage=0.98):
    """
    transform to uint8
    image shape (c, h, w)
    """
    img_ds = gdal.Open(image_path)
    img_geo = img_ds.GetGeoTransform()
    img_proj = img_ds.GetProjection()

    image = img_ds.ReadAsArray()  # (c, h, w)
    del img_ds
    C, H, W = image.shape
    print('image', image.shape, np.max(image), image.dtype)

    image = np.nan_to_num(image, nan=0)  # transform nan to 0
    img_list = np.split(image, C, axis=0)
    min_list = [np.quantile(r[r != 0], min_percentage) for r in img_list]
    max_list = [np.quantile(r[r != 0], max_percentage) for r in img_list]
    # print(min_list, max_list)

    image_bands = [image_stretch_2d(image[i, :, :], min_list[i], max_list[i]) for i in range(C)]
    image_new = np.stack(image_bands, axis=0)

    project_array_and_save(image_new, save_path, img_geo, img_proj)
