
# Music Detection and Music Relative Loudness Estimation

This is the official PyTorch implementation of *Bijue Jia, et al. "Hierarchical Regulated Iterative Network for Joint Task of Music Detection and Music Relative Loudness Estimation"([paper](https://ieeexplore.ieee.org/document/9222227)).

In the earliest version of our paper, we call our model HEDLN. It is substituted by HRIN in the newest manuscript. However, we haven't changed the abbreviation in the code, so don't get confused.   

## Requirements
- `PyTorch` for the deep learning implementations.
- `librosa` for the pre-processing of the audio. ([link](https://github.com/librosa/librosa "librosa"))
- `dcase_util` A collection of utilities for Detection and Classification of Acoustic Scenes and Events. ([link](https://github.com/DCASE-REPO/dcase_util "dcase_util"))
- `sed_eval` for the evaluation of the models. ([link](https://github.com/TUT-ARG/sed_eval "sed_eval"))
- `sed_vis` for visualizing the annotations. ([link](https://github.com/TUT-ARG/sed_vis "sed_vis"))
- `pandas`

## Data
OpenBMAT dataset ([link](https://zenodo.org/record/3381249 "OpenBMAT"))

## Attribution
If you use this work in your research, please cite us via the following BibTeX:

```
@article{jia2020hierarchical,
  title={Hierarchical Regulated Iterative Network for Joint Task of Music Detection and Music Relative Loudness Estimation},
  author={Jia, Bijue and Lv, Jiancheng and Peng, Xi and Chen, Yao and Yang, Shenglan},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={1--13},
  year={2020},
  publisher={IEEE}
}
```

