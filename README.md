# Adversarial Drift Compensation [![Paper](https://img.shields.io/badge/arXiv-2210.07207-brightgreen)](https://arxiv.org/pdf/2405.19074)
## Code for CVPR 2024 paper - Resurrecting Old Classes with New Data for Exemplar-Free Continual Learning

## Abstract
Continual learning methods are known to suffer from catastrophic forgetting, a phenomenon that is particularly hard to counter for methods that do not store exemplars of previous tasks. Therefore, to reduce potential drift in the feature extractor, existing exemplar-free methods are typically evaluated in settings where the first task is significantly larger than subsequent tasks. Their performance drops drastically in more challenging settings starting with a smaller first task. To address this problem of feature drift estimation for exemplar-free methods, we propose to adversarially perturb the current samples such that their embeddings are close to the old class prototypes in the old model embedding space. We then estimate the drift in the embedding space from the old to the new model using the perturbed images and compensate the prototypes accordingly. We exploit the fact that adversarial samples are transferable from the old to the new feature space in a continual learning setting. The generation of these images is simple and computationally cheap. We demonstrate in our experiments that the proposed approach better tracks the movement of prototypes in embedding space and outperforms existing methods on several standard continual learning benchmarks as well as on fine-grained datasets.

<img src="https://github.com/dipamgoswami/ADC/blob/main/figs/ADC.png" width="100%" height="100%">

```
@inproceedings{goswami2024resurrecting,
  title={Resurrecting Old Classes with New Data for Exemplar-Free Continual Learning},
  author={Goswami, Dipam and Soutif-Cormerais, Albin and Liu, Yuyang and Kamath, Sandesh and Twardowski, Bart{\l}omiej and van de Weijer, Joost},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Algorithm
<p align="center">
<img src="https://github.com/dipamgoswami/ADC/blob/main/figs/algo.png" width="50%" height="100%">
</p>

## Results
<img src="https://github.com/dipamgoswami/ADC/blob/main/figs/results.png" width="100%" height="100%">

## Implementation

The code framework is taken from [PyCIL](https://github.com/G-U-N/PyCIL).

### Dependencies
1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)

### Datasets

We performed experiments for `CIFAR100`, `ImageNet100,`, `TinyImageNet`, `CUB200` and `Stanford Cars`. When training on `CIFAR100`, this framework will automatically download it.  When training on other datasets, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```
To download ImageNet-Subset dataset: [Link](https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn)

### Run experiments

The code for ADC can be found in [models/lwf.py](https://github.com/dipamgoswami/ADC/blob/main/models/lwf.py).

To use ADC, run

   ```
    python main.py --config=exps/lwf.json
   ```

The configs can be modified in [exps/lwf.json](https://github.com/dipamgoswami/ADC/blob/main/exps/lwf.json).

