# EH-SurGS: Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes

Official code implementation for [EH-SurGS](https://arxiv.org/abs/2501.01101), a Gaussian Splatting based framework for surgical scene reconstruction.

<!--### [Project Page]() -->

> [EH-SurGS: Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes](https://arxiv.org/abs/2501.01101)\
> Jiwei Shan, Zeyu Cai, Cheng-Tai Hsieh, Shing Shin Cheng, Hesheng Wang\
> ICRA2025, **Accept**


## Environment setup

Tested with NVIDIA RTX 4090 GPU.

```bash
git clone https://github.com/IRMVLab/EH-SurGS.git
cd EH-SurGS
conda create -n EH-SurGS python=3.7 
conda activate EH-SurGS

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Datasets

We use [EndoNeRF](https://github.com/med-air/EndoNeRF), [StereoMIS](https://zenodo.org/records/7727692) and processed [Hamlyn](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion) datasets to verify our method.

The data structure is as follows:

```
data
| - endonerf_full_datasets
|   | - cutting
|   |   | -  depth/
|   |   | -  images/
|   |   | -  masks/
|   |   | -  pose_bounds.npy 
|   | - pushing
|   |   | -  depth/
|   |   | -  images/
|   |   | -  masks/
|   |   | -  pose_bounds.npy 
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

## Citation

If you find this code useful for your research, please use the following BibTeX entries:

```
@article{shan2025deformable,
      title={Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes},
      author={Shan, Jiwei and Cai, Zeyu and Hsieh, Cheng-Tai and Cheng, Shing Shin and Wang, Hesheng},
      journal={arXiv preprint arXiv:2501.01101},
      year={2025}
}

```

### Questions

For further question about the code or paper, welcome to create an issue, or contact 'jiweishan@link.cuhk.edu.hk' or 'caizeyu1106@sjtu.edu.cn'
