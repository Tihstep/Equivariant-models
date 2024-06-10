
Equivariant Convolutional Neural Networks in Metamaterial Domain
--------------------------------------------------------------------------------
**[Article](https://arxiv.org/)** | **[Experiments](https://api.wandb.ai/links/tihstepml/54mw7gv0)**

--------------------------------------------------------------------------------
**The experiments conducted in this work demonstrate the importance of incorporating appropriate equivariance properties into the model architecture when dealing with data that exhibits specific symmetries.**

By leveraging the a priori knowledge about the nature of the metamaterials dataset, we were able to design models that are equivariant to the relevant transformations, such as reflections and circular shifts. These equivariant models were shown to converge faster and achieve better performance compared to standard, non-equivariant models, especially when the training data was augmented with the corresponding transformations.


--------------------------------------------------------------------------------
*Equivariant neural networks* guarantee a specified transformation behavior of their feature spaces under transformations of their input.
For instance, classical convolutional neural networks (*CNN*s) are by design equivariant to translations of their input.
This means that a translation of an image leads to a corresponding translation of the network's feature maps.
This package provides implementations of neural network modules which are equivariant under all *isometries* E(2) of the image plane $\mathbb{R}^2$, that is, under *translations*, *rotations* and *reflections*.
In contrast to conventional CNNs, E(2)-equivariant models are guaranteed to generalize over such transformations, and are therefore more data efficient.

The feature spaces of E(2)-Equivariant Steerable CNNs are defined as spaces of *feature fields*, being characterized by their transformation law under rotations and reflections.
Typical examples are scalar fields (e.g. gray-scale images or temperature fields) or vector fields (e.g. optical flow or electromagnetic fields).

![feature field examples](https://github.com/Tihstep/Equivariant-models/blob/main/data/feature_fields.png)
(image taken from Maurice Weiler e2cnn)

--------------------------------------------------------------------------------

The library is structured into five subpackages with different high-level features:

| Component                                                                              | Description                                                      |
| ---------------------------------------------------------------------------------------| ---------------------------------------------------------------- |
| [**Equivariant-models.KR**](https://github.com/Tihstep/Equivariant-models/blob/main/KR.ipynb)          | repository entry point, runs all experiments |
| [**Equivariant-models.models**](https://github.com/Tihstep/Equivariant-models/tree/main/models)      | folder with all model architectures  |
| [**Equivariant-models.data**](https://github.com/Tihstep/Equivariant-models/tree/main/data)      | folder with data and visualization |
| [**Equivariant-models.data_config.yaml**](https://github.com/Tihstep/Equivariant-models/blob/main/data_config.yaml)      |  data_configuration  |
| [**Equivariant-models.model_config.yaml**](https://github.com/Tihstep/Equivariant-models/blob/main/model_config.yaml)                | models configuration |
---------------------------------------------------------------------------------------------------------------------------------------------------

## Experimental results

E(2)-steerable convolutions can be used as a drop in replacement for the conventional convolutions used in CNNs.
Keeping the same training setup and *without performing hyperparameter tuning*, this leads to significant performance boosts compared to CNN baselines (values are test errors in percent):

 model                 | Non-augmented data   | Flip              | Rotation         | Circular Shift   |
 ----------------------| ---------------------| ------------------|------------------|------------------|
 CNN baseline          | 0.01   ± 0.004       | 0.011 ± 0.004     | 0.007 ± 0.003    | 0.011 ± 0.004    |
 FLip Equivariant      | 0.014  ± 0.006       | 0.008 ± 0.001     |                  |                  |
 Rotation Equivariant  | 0.0143 ± 0.005       |                   | 0.005 ± 0.001    |                  |
 Shift Equivariant     | 0.0193 ± 0.008       |                   |                  | 0.01 ± 0.40      |


 <img src="https://github.com/Tihstep/Equivariant-models/blob/main/data/Non_aug.png" width="700" height="350">
 <img src="https://github.com/Tihstep/Equivariant-models/blob/main/data/Flip.png" width="700" height="350">
 <img src="https://github.com/Tihstep/Equivariant-models/blob/main/data/Shift.png" width="700" height="350">
 <img src="https://github.com/Tihstep/Equivariant-models/blob/main/data/Rotation.png" width="700" height="350">


## Dependencies

The library is based on Python3.7

```
torch>=1.1
numpy
scipy
```
Optional:
```
pymanopt
autograd
```

The following packages
```
sympy
rbf
```
are required to use the steerable differential operators.
