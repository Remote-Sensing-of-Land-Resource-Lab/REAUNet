import os
from osgeo import gdal, ogr, osr


def tif2shp(result_path, out_shp_path):
    in_raster = gdal.Open(result_path)
    in_band = in_raster.GetRasterBand(1)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(in_raster.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_shp_path):  # delete shp
        drv.DeleteDataSource(out_shp_path)

    Polygon1 = drv.CreateDataSource(out_shp_path)  # create the target shp
    Poly_layer = Polygon1.CreateLayer(
        result_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon
    )  # create a polygon layer
    newField = ogr.FieldDefn("value", ogr.OFTReal)  # add a field to save pixel value
    Poly_layer.CreateField(newField)

    gdal.Polygonize(in_band, None, Poly_layer, 0)  # tif to shp
    # gdal.FPolygonize(inband, None, Poly_layer, 0)

    for feature in Poly_layer:
        if feature.GetField("value") == 0:
            Poly_layer.DeleteFeature(feature.GetFID())  # delete 0
    Polygon1.SyncToDisk()
    Polygon1 = None
    print('Tif to Shape')
