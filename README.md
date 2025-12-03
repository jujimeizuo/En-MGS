

## Environment setup

Tested with NVIDIA RTX 4090 GPU.

```bash
git clone https://github.com/IRMVLab/EH-SurGS.git
cd EH-SurGS
conda create -n EH-SurGS python=3.10
conda activate EH-SurGS

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Datasets

We use [EndoNeRF](https://github.com/med-air/EndoNeRF), [StereoMIS](https://zenodo.org/records/7727692) and processed [Hamlyn](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion) datasets to verify our method.

To use the two available examples in [EndoNeRF](https://github.com/med-air/EndoNeRF) dataset. Please download the data via [this link](https://forms.gle/1VAqDJTEgZduD6157) and organize the data according to the [guideline](https://github.com/med-air/EndoNeRF.git).

To use the [StereoMIS](https://zenodo.org/records/7727692) dataset, please follow this [github repo](https://github.com/aimi-lab/robust-pose-estimator) to preprocess the dataset. After that, run the provided script `stereomis2endonerf.py` to extract clips from the StereoMIS dataset and organize the depth, masks, images, intrinsic and extrinsic parameters in the same format as [EndoNeRF](https://github.com/med-air/EndoNeRF). In our implementation, we used [RAFT](https://github.com/princeton-vl/RAFT) to estimate the stereo depth for [StereoMIS](https://zenodo.org/records/7727692) clips. Following EndoNeRF dataset, this script only supports fixed-view settings.

The data structure is as follows:

```
data
| - endonerf_full_datasets
|   | - cutting_tissues_twice
|   |   | -  depth/
|   |   | -  images/
|   |   | -  masks/
|   |   | -  pose_bounds.npy 
|   | - pushing_soft_tissues
| - StereoMIS
|   | - stereo_seq_1
|   | - stereo_seq_2
```
## Scrpits

```bash
python stereomis2endonerf.py -d ../Datasets/StereoMIS/P3 -f $start $end
```

```bash
bash run_pipeline.sh [train | render | metric | all] [dataset_path] [metric_param (option)]"
```

## Training

To train EH-SurGS with customized hyper-parameters, please make changes in `arguments/endonerf/default.py`.

To train EH-SurGS, run the following example command:

```
python train.py -s data/endonerf/cutting --expname endonerf/cutting --configs arguments/endonerf/default.py 
```

## Testing

For testing, we perform rendering and evaluation separately.

### Rendering

To run the following example command to render the images:

```
python render.py --model_path output/endonerf/cutting  --skip_train --configs arguments/endonerf/default.py
```

### Evaluation

To evaluate the reconstruction quality, run following command:

```
python metrics.py --model_path output/endonerf/cutting -p test
```

Note that you can set `-p video`, `-p test`, `-p train` to select the set for evaluation.

## Acknowledgements

This repo borrows some source code from [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master), [4DGS](https://github.com/hustvl/4DGaussians), [Deform3DGS](https://github.com/jinlab-imvr/Deform3DGS), [depth-diff-gaussian-rasterizer](https://github.com/ingra14m/depth-diff-gaussian-rasterization), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and [EndoNeRF](https://github.com/med-air/EndoNeRF). We would like to acknowledge these great prior literatures for inspiring our work.

Thanks to [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master) for their great and timely effort in releasing the framework adapting Gaussian Splatting into surgical scene.

