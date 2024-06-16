from osgeo import gdal
from skimage import io, morphology
import numpy as np
from data_crop.project_array import project_array_and_save


def cutoff(extent_path,
           boundary_path,
           result_path,
           threshold: float = 0.5):
    """
    extent_map & boundary_map: float (0-1)
    """
    extent_ds = gdal.Open(extent_path)
    geot = extent_ds.GetGeoTransform()
    proj = extent_ds.GetProjection()
    del extent_ds

    extent_map = io.imread(extent_path)
    boundary_map = io.imread(boundary_path)
    print('extent map:', extent_map.shape, np.max(extent_map))
    print('boundary map:', boundary_map.shape, np.max(boundary_map))

    extent = np.where(extent_map > threshold, 1, 0)
    extent = extent.astype(np.uint8)
    boundary = np.where(boundary_map > threshold, 1, 0)
    boundary = boundary.astype(np.uint8)

    boundary = boundary.astype(np.bool_)
    boundary = morphology.dilation(boundary, footprint=morphology.square(2))
    boundary = morphology.thin(boundary)
    boundary = morphology.remove_small_objects(boundary, min_size=10, connectivity=2)
    boundary = boundary.astype(np.uint8)

    result = extent - boundary
    result = np.where(result == 1, 1, 0)

    result = result.astype(np.bool_)
    result = morphology.remove_small_objects(result, min_size=10, connectivity=1)
    result = morphology.remove_small_holes(result, area_threshold=10, connectivity=1)
    result = result.astype(np.uint8)

    result = (result * 255).astype(np.uint8)
    project_array_and_save(result, result_path, geot, proj)
