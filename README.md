# REAUNet

Pytorch implement of the REAUNet for agricultural parcel delineation. 

## Paper

A Refined Edge-Aware Convolutional Neural Network for Agricultural Parcel Delineation (under review)

## Architecture

![](https://github.com/Remote-Sensing-of-Land-Resource-Lab/REAUNet/blob/main/figures/figure1.jpg)

## Experiment Results

![](https://github.com/Remote-Sensing-of-Land-Resource-Lab/REAUNet/blob/main/figures/figure2.jpg)

## Samples

Prepare the remote sensing images and corresponding labels required for training, both in `.tif` format. Use `data_crop.py` in the `data_crop` directory to generate the training samples, and then use `data_txt.py` to generate a `.txt` file containing the names of the training samples. 

## Training

Run `train.py` to train the model. The `config.py` file contains the configuration parameters for training, and you can set your own training parameters as needed.  Noted that in `config.py`, there are four parameters required for training that need to be specified: 

```python
parser.add_argument('--txt_path_train', type=str, required=True, help='path to the training text file')
parser.add_argument('--txt_path_val', type=str, required=True, help='path to the valid text file')
parser.add_argument('--image_root', type=str, required=True, help='file folder of the training samples')
parser.add_argument('--label_root', type=str, required=True, help='file folder of the labels')
```

These parameters can be specified in two ways. First, you can add the parameter paths in `config.py` and remove the `required` settings. Second, you can provide the parameter paths when runing `train.py`. The methods are as follows: 

```python
parser.add_argument('--txt_path_train', type=str, default=r'DATA_TRAIN.txt', help='path to the training text file')
parser.add_argument('--txt_path_val', type=str, default=r'DATA_VAL.txt', help='path to the valid text file')
parser.add_argument('--image_root', type=str, default=r'D:\DATA\IMAGE', help='file folder of the training samples')
parser.add_argument('--label_root', type=str, default=r'D:\DATA\LABEL', help='file folder of the labels')
```

```python
python train.py --txt_path_train=DATA_TRAIN.txt --txt_path_val=DATA_VAL.txt --image_root=D:\DATA\IMAGE --label_root=D:\DATA\LABEL
```

## Predict

The trained model weights file is saved in the automatically generated `checkpoint` folder. Load the model weights file and run `predict.py` to get the prediction results. Note that the input remote sensing image file needs to be in `uint8` format. If it is in `float` format, you can use `image_stretch.py` in the `data_crop` directory to convert it. 

## Post-processing

The post-processing of this method requires the input of the model-extracted agricultural parcel extent output and agricultural parcel boundary output. First, use `cutoff.py` to fuse the two results and obtain the final parcel results. Then, use `tif2shp.py` to convert the raster results into vector results. Finally, use `shp_refine.py` to optimize the vector parcels. 

## Contributors

Rui Lu (lurui98@zju.edu.cn), Su Ye
