import numpy as np
from osgeo import gdal


def project_array_and_save(array, save_path, geotransform, projection):
    """
    array.shape: (c, h, w) or (h, w)
    Usage:
        image_ds = gdal.Open(image_path)
        geotrans = image_ds.GetGeoTransform()
        proj = image_ds.GetProjection()
        # del image_ds
        ...
        copy_proj(image, save_path, geotrans, proj)
    """
    if 'int8' in array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(array.shape) == 3:
        bands, height, width = array.shape
    else:
        bands, (height, width) = 1, array.shape

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(save_path, width, height, bands, datatype)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    if bands > 3:
        # Warning 1: ...
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_GrayIndex)  # no alpha

    if bands == 1:
        if len(array.shape) == 3:
            array = np.squeeze(array, axis=0)
        ds.GetRasterBand(1).WriteArray(array)
    else:
        for band in range(bands):
            ds.GetRasterBand(band + 1).WriteArray(array[band])
