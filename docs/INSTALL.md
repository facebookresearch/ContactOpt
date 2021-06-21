# Installation

As some of the dependencies for this project are non-trivial to install, the installation instructions are provided in this document. It is not necessary to follow these instructions exactly as long as all the necessary packages are installed.

#### Environment Setup

We recommend using a new conda environment.
```
conda create -n contactopt python=3.8
conda activate contactopt
```

#### Install PyTorch and PyTorch3D
Detailed [installation instructions](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) can be found on the project website. Note that Pytorch3D places restrictions on the versions of Python and PyTorch used.
```
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

#### Install PyTorch-Geometric
Detailed [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) can be found on the project website.

To ensure that the package has been installed correctly, test that the following snippet runs without error:
```
python -c "import torch; assert torch.cuda.is_available(); import torch_geometric.data"
```

#### Install other dependencies
Other dependencies play nicely with pip and can be installed with:
```
pip install git+https://github.com/hassony2/manopth.git open3d tensorboardX pyquaternion trimesh transforms3d chumpy opencv-python
```

#### Download MANO Model
Download the Python 3-compatible MANO code from the [manopth website](https://github.com/hassony2/manopth). Copy the `mano` folder from the manopth project to the root of the ContactOpt folder.

Due to license restrictions, the MANO data files must be downloaded from the original [project website](https://mano.is.tue.mpg.de/). Create an account and download 'Models & Code'. Extract the `models` folder to the recently created `mano` directory. The directory structure should be arranged so the following files can be found:
```
mano/webuser/lbs.py
mano/models/MANO_RIGHT.pkl
```

## Download and Install ContactPose (optional)

This step is required to interact with the ContactPose or Perturbed ContactPose dataset. This is a requirement for retraining the DeepContact network. Note that the Perturbed ContactPose data file is large (~40 GB) and requires a computer with ~64 GB of RAM.

[Installation instructions](https://github.com/facebookresearch/ContactPose/blob/master/docs/doc.md) can be found on the project website.

Edit the line at the top of `contactopt/create_dataset_contactpose.py` to point to the recently installed ContactPose directory.
```
sys.path.append('../ContactPose')   # Change this path to point to the ContactPose repo
```

To generate the dataset files for Perturbed ContactPose, execute the following script. This may take up to an hour to complete.

```
python contactopt/create_dataset_contactpose.py 
```

## Download Image Pose Estimates from the HO-3D dataset (optional)

To evaluate the performance of ContactOpt on the results of a [RGB pose estimator](https://arxiv.org/abs/2004.13449) on the [HO-3D dataset](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/), download the pre-generated pose estimates file [[mirror 1]](https://www.dropbox.com/s/349xj3bu446mmgw/pose_estimates.pkl?dl=0) (6 GB). Place it in the `data` folder.

The dataset file can be generated with:
```
python contactopt/create_dataset_im.py 
```