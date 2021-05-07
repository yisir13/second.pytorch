## Getting Started

source: https://github.com/nutonomy/second.pytorch
If you want to see how to train KITTI Dataset, please reference it.

### Code Support

My environment: python3.7/3.8 pytorch=1.7 Ubuntu18.04

### Install

#### 1. Clone code

```bash
git clone https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet/
git clone https://gitlab.uni-hannover.de/yisirli/pedestrian_pointpillars-private.git
bash build.sh
```

#### 2. Install Python packages

It is recommend to use the Anaconda package manager.

First, use Anaconda to configure as many packages as possible.
```bash
conda create -n pointpillars python=3.7 anaconda
source activate pointpillars

conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision
conda install sparsehash 

pip install fire tensorboardX pypcd
sudo apt-get update
sudo apt-get install libboost-all-dev
```

#### 3. Setup cuda for numba

You need to add following environment variables for numba to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

#### 4. PYTHONPATH

Add second.pytorch/ to your PYTHONPATH.
```bash
gedit ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/home/SparseConvNet/Pedestrian_PointPillars '>>.bashrc
source ~/.bashrc
```

### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── second/dataset_root
       ├── training    <--  data for evaluation
       |   ├── label_2
       |   ├── velodyne

```


#### 2. Create kitti infos:

```bash
python create_vlp16.py create_ikg_info_file --data_path=./data_vlp16/object --reduce=False
```


#### 3. Modify config file

The config file needs to be edited to point to the above datasets:

```bash

eval_input_reader: {
  ...
  kitti_info_path: "/path/to/ikg_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```




### Evaluate


```bash
cd ~/Pedestrian_PointPillars/second/
pytorch/evaluation_ikg.py run in spyder
```


