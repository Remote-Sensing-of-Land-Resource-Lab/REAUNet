import geopandas as gpd
from shapely.geometry import Polygon
import math


def eliminate_spike(polygon,
                    max_angle_deg=45.0):
    max_angle_rad = math.radians(max_angle_deg)
    # print(max_angle_rad)

    coords = list(polygon.exterior.coords)

    coords.append(coords[0])
    coords.append(coords[1])
    new_coords = []

    for i in range(1, len(coords) - 1):
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[i + 1]

        # math.atan2(y, x) -- atan(y/x) out value: (-pi, pi)
        angle = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])

        if angle < 0:
            angle += 2 * math.pi

        if max_angle_rad <= angle <= 2 * math.pi - max_angle_rad:
            new_coords.append(p2)

    if len(new_coords) >= 4:
        new_polygon = Polygon(new_coords)
        area1 = new_polygon.area
        area2 = polygon.area
        if 1.2 >= (area1 / area2) >= 0.8:
            return new_polygon
        else:
            return polygon
    else:
        return polygon


def simplify_shp(out_shp_path,
                 save_shp_path,
                 t: float = 1,
                 degree: float = 30.0):
    """
    Douglas-Peucker
    """
    raw_gdf = gpd.read_file(out_shp_path)
    gdf = raw_gdf.copy()

    print("Douglas-Peucker -- tolerance:", t)

    for index, row in gdf.iterrows():
        simplified_polygon = row['geometry'].simplify(tolerance=t)
        simplified_polygon = eliminate_spike(simplified_polygon, degree)
        gdf.at[index, 'geometry'] = simplified_polygon

    num_features = len(gdf)
    total_nodes = gdf.geometry.apply(lambda geom: len(list(geom.exterior.coords))).sum()
    if total_nodes > 10 * num_features:
        print(f"Too many nodes！  parcels: {num_features}   nodes: {total_nodes}]")
    else:
        print(f"parcels: {num_features}   nodes: {total_nodes}")

    gdf.to_file(save_shp_path, driver="ESRI Shapefile")
