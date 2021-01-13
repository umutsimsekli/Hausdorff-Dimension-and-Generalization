# Hausdorff Dimension and Generalization

This code repository includes the source code for the [Paper](https://arxiv.org/abs/2006.09313):

```
Hausdorff Dimension, Heavy Tails, and Generalization in Neural Networks
Umut Simsekli, Ozan Sener, George Deligiannidis, Murat A Erdogdu
Advances in Neural Information Processing Systems (NeurIPS) 2020 
```

The source code is released under the MIT License. See the License file for details.


# Usage
Since the project is about generalization, it requires training lots of models. Hence, sharing a single executable is not reasonable as execution depends on the cluster structure. Hence, we share all the source code which needs to be used by the right scripts for the cluster manager. A basic pipeline for VGG experiment is:

- Running the training script which calls `python train_vgg.py` which will save models to `./models/' folder.
- Running the estimator via: ```python alpha_estimator_vgg.py --model_folder=folder```
- Evalaute the test accuracies via: ```python eval_vgg.py --model_folder=folder```


# Source Code Explanations
- `train_lenet.py`: Trains a collection of LeNet type networks with various number of layers for given learning rate and batch size over CIFAR using SGD. All trained models are saved in `./models/` folder. We save a model every 5 epoch until the 50th one. Then, we save all models for each batch since it is used for estimation of  Blumenthal-Getoor Index.
- `train_vgg.py`: Trains a collection of VGG type networks with various number of layers for given learning rate and batch size over CIFAR using SGD. All trained models are saved in `./models/` folder. We save a model every 20 epoch until the 80th one. Then, we save all models for each batch since it is used for estimation of  Blumenthal-Getoor Index.
- `eval_lenet.py`: Evaluate collection of trained LeNet models on the test set of the CIFAR.
- `eval_vgg.py`: Evaluate colleciton of trained VGG models on the test set of the CIFAR.
- `alpha.py`: Collection of Blumenthal-Getoor Index estimators. We only use `estimator_vector_projected` function but others are included in case it is useful for future research.
- `alpha_estimator_vgg.py`: Estimate the Blumenthal-Getoor Index of the trained models.
- `models.py`: Collection of VGG-style and LeNet-style neural networks with different number of layers.


# Citation
If you use this codebase or any part of it for a publication, please cite:
```
@inproceedings{NeurIPS2020_Hausdorff,
title={Hausdorff Dimension, Heavy Tails, and Generalization in Neural Networks
},
author={Umut Simsekli, Ozan Sener, George Deligiannidis, Murat A Erdogdu
},
booktitle={Advances in Neural Information Processing Systems},
year={2020}
}
```