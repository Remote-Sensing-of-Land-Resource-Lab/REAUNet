import math
from osgeo import gdal
from skimage import io
import numpy as np
import time
import torch

from model import REAUNet
from data_crop.project_array import project_array_and_save


def model_output(image, model, device):
    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(image)
    if isinstance(out, list):
        out = out[-1]
    if out.device != 'cpu':
        out = out.cpu()
    return out


def model_predict(image_path,
                  save_path,
                  in_channels: int,
                  num_class: int,
                  checkpoint_path,
                  device='cuda:0' if torch.cuda.is_available() else 'cpu',
                  batch_size: int = 8,
                  crop_size: int = 256,
                  crop_step: int = 256):
    # load model
    model = REAUNet(in_channels, num_class)
    model.load_state_dict(torch.load(checkpoint_path))

    print('=' * 20, 'Predict', '=' * 20)
    image_ds = gdal.Open(image_path)
    proj = image_ds.GetProjection()
    geotrans = image_ds.GetGeoTransform()
    height, width = image_ds.RasterYSize, image_ds.RasterXSize
    rows = math.ceil((height - crop_size) / crop_step + 1)
    cols = math.ceil((width - crop_size) / crop_step + 1)
    del image_ds
    print(f'image.shape ({height:d}, {width:d})  rows%cols ({rows:d}, {cols:d})  batch size: {batch_size:d}')

    # read image
    image = io.imread(image_path) / 255.  # (h, w, c)

    image_results = np.zeros(shape=(height, width), dtype=np.float32)  # result map
    image_count = np.zeros(shape=(height, width), dtype=np.float32)  # pixel count
    count = np.ones(shape=(crop_size, crop_size), dtype=np.float32)

    time1 = time.perf_counter()
    batch = 0
    sub_image_list = []  # sub images
    sub_image_location = []  # location information

    for i in range(rows):
        for j in range(cols):
            if i != rows - 1 and j != cols - 1:
                image_sub = image[i * crop_step:i * crop_step + crop_size, j * crop_step:j * crop_step + crop_size, :]
            elif i != rows - 1 and j == cols - 1:
                image_sub = image[i * crop_step:i * crop_step + crop_size, width - crop_size:width, :]
            elif i == rows - 1 and j != cols - 1:
                image_sub = image[height - crop_size:height, j * crop_step:j * crop_step + crop_size, :]
            else:
                image_sub = image[height - crop_size:height, width - crop_size:width, :]

            if np.sum(image_sub) == 0:
                continue

            sub_image = torch.from_numpy(image_sub).float()  # to tensor
            sub_image = sub_image.permute(2, 0, 1).unsqueeze(0)
            # print(sub_image.shape, torch.max(sub_image), i, j)
            sub_image_list.append(sub_image)  # image
            sub_image_location.append((i, j))  # location

            print('\r' + f'[{i * cols + j + 1:d}/{rows * cols:d}] {sub_image.shape:s} ({i + 1:d}, {j + 1:d})')

            if (batch + 1) % batch_size == 0 or ((i == (rows - 1)) & (j == (cols - 1))):
                model_input = torch.cat(sub_image_list, dim=0)
                batch_out = model_output(model_input, model, device)
                batch_out = batch_out.squeeze(1).numpy()

                for b in range(batch_out.shape[0]):
                    batch_image = batch_out[b, :, :, :]
                    batch_location = sub_image_location[b]
                    h, w = int(batch_location[0]), int(batch_location[1])

                    # print('\r', b, h, w, batch_image.shape, end='')
                    if h != rows - 1 and w != cols - 1:
                        image_results[h * crop_step:h * crop_step + crop_size,
                                      w * crop_step:w * crop_step + crop_size] += batch_image[0, :, :]
                        image_count[h * crop_step:h * crop_step + crop_size,
                                    w * crop_step:w * crop_step + crop_size] += count
                    elif h != rows - 1 and w == cols - 1:
                        image_results[h * crop_step:h * crop_step + crop_size,
                                      width - crop_size:width] += batch_image[0, :, :]
                        image_count[h * crop_step:h * crop_step + crop_size,
                                    width - crop_size:width] += count
                    elif h == rows - 1 and w != cols - 1:
                        image_results[height - crop_size:height,
                                      w * crop_step:w * crop_step + crop_size] += batch_image[0, :, :]
                        image_count[height - crop_size:height,
                                    w * crop_step:w * crop_step + crop_size] += count
                    else:
                        image_results[height - crop_size:height, width - crop_size:width] += batch_image[0, :, :]
                        image_count[height - crop_size:height, width - crop_size:width] += count

                if (batch + 1) % batch_size == 0:
                    sub_image_list = []
                    sub_image_location = []
                    batch = 0
            else:
                batch += 1

    image_count = np.where(image_count == 0, 1, image_count)
    result = image_results / image_count
    result = result.astype(np.float32)
    print('\r' + 'result.shape:', result.shape, '[time:%.2fs]' % (time.perf_counter() - time1))

    project_array_and_save(result, save_path, geotrans, proj)
