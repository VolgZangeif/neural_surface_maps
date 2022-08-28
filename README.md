# Neural Surface Maps

Official implementation of **Neural Surface Maps** - [Luca Morreale](https://luca-morreale.github.io/), [Noam Aigerman](https://noamaig.github.io/), [Vladimir Kim](http://www.vovakim.com/), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/)

[[Paper]](https://arxiv.org/abs/2103.16942) [[Project Page]](http://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/)

## How-To

Replicating the results is possible following these steps:
1. Parametrize the surface
2. Prepare surface sample
3. Overfit the surface
4. Neural parametrization of the surface
5. Optimize surface-to-surface map
6. Optimize a map between a collection

#### 1. Surface Parametrization
This is a preprocessing step. You can use SLIM[1] from [this repo](https://github.com/luca-morreale/simple_parametrization) to fulfill this step.

#### 2. Sample preparation
Given a parametrized surface (prev. step), we need to convert it into a sample. First of all, we need to over sample the surface with Meshlab. You can use the midpoint subdivision filter.

Once the super-sampled surface is ready then you can convert it into a sample:
```py
python -m preprocessing.convert_sample surface_slim.obj surface_slim_oversampled.obj output_sample.pth
```
The file `output_sample.pth` is the sample ready to be over-fitted.

#### 3. Overfit surface
A surface representation is generated with:
```py
python -m train_surface_map dataset.sample_path=output_sample.pth
```

Other way to do it is to set the sample path in the config file and place it in the folder `experiments/surface_map` and then use the command

```py
python -m train_surface_map +surface_map=base
```

This reads the config file `experiments/surface_map/base.yaml`

This will save a surface map inside `outputs/neural_maps` folder.
The folder name follows this patterns: `overfit_[suffix]`.
Inside that folder, the map is saved under the `sample` folder as `pth` file.
The suffix is used as given in the config file.

**NOTE** that this assumes the existence of a folder called output in the base directory

The overfitted surface can be generated with:
```py
python -m show_surface_map
```
please, set the path to the `pth` file just created inside the script.

#### 4. Neural parametrization
Generating a neural parametrization need to run:
```py
python -m train_parametrization_map dataset.sample_path=your_surface_map.pth
```
Like for the overfitting, this saves the map inside `outputs/neural_maps` folder. The folder name have the following patterns `parametrization_[timestamp]`.

To display the paramtrization obtained run:
```py
python -m show_parametrization_map
```
please, set the path to the `pth` file just created inside the script.

#### 5. Optimize surface-to-surface map
To generating a inter-surface map run:
```py
python -m train_intersurface_map dataset.sample_path_g=your_surface_map_a.pth dataset.sample_path_f=your_surface_map_b.pth
```

Other way to do it is to set the sample path in the config file and place it in the folder `experiments/inter_map` and then use the command

```py
python -m train_intersurface_map +inter_map=base
```

This reads the config file `experiments/inter_map/base.yaml`



Note, this steps requires two surface maps. A source, `sample_path_g`, and a target, `sample_path_f`.
Further, it requires the landmark points for the 2 surfaces.
Remember to set them in the config file or on command line before running the program.

Likewise the overfitting, the map is saved inside `outputs/neural_maps`.
The inter-surface map folder pattern is `intersurface_[suffix]`.
The `pth` file is inside the `models` folder.
The suffix is used as given in the config file.


To display the inter-surface map run:
```py
python -m show_intersurface_map
```
remember to set the path of the maps inside the script.
Also, remember to specify the landmark points in this script as well.


#### 6. Optimize collection map
A collection between a set of surface maps can be optimized with:
```py
python -m train_intersurface_map dataset.sample_path_g=your_surface_map_g.pth dataset.sample_path_f=your_surface_map_f.pth dataset.sample_path_q=your_surface_map_q.pth
```
Note, this steps requires three surface maps. A source, `sample_path_g`, and two targets, `sample_path_f` and `sample_path_q`.

This will save two maps inside `outputs/neural_maps` folder. The folder name follows this patterns: `collection_[timestamp]`, under the folder `models` you can find two `*.pth` file.

To display the collection map run:
```py
python -m show_collection_map
```
remember to set the path of maps inside the script.

---

## Dependencies

Dependencies are listed in `environment.yml`. Using conda, all the packages can be installed with `conda env create -f environment.yml`.

On top of the packages above, please install also [pytorch svd on gpu](https://github.com/KinglittleQ/torch-batch-svd) package.

---

## Data

Any mesh can be used for this process. A data example can be downloaded [here](https://mega.nz/folder/SPQXkI6D#Wuq86POJlCgZUl3i7ueVow).



---
## Citation
```
@misc{morreale2021neural,
      title={Neural Surface Maps},
      author={Luca Morreale and Noam Aigerman and Vladimir Kim and Niloy J. Mitra},
      year={2021},
      eprint={2103.16942},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
---
## References
[1] Scalable locally injective mappings - Michael Rabinovich *et. al.* - ACM Transactions on Graphics (TOG) 2017


