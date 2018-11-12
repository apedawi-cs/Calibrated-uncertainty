# Calibrated uncertainty
Playing around with ideas from a few papers:

[1] [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/pdf/1612.01474.pdf)

[2] [Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors](https://arxiv.org/pdf/1807.09289v2.pdf)

Run with `python3 example.py`.

### Required Python 3.6.6 dependencies:
```
matplotlib==3.0.0
numpy==1.15.2
tensorflow==1.12.0
tensorflow_probability==0.5.0
```

---------

### Results from toy experiment

#### - Data generating process:

<img src="https://github.com/apedawi-cs/Calibrated-uncertainty/blob/master/dgp.png" width="300">

#### - Actual vs. estimated log density surface:

<img src="https://github.com/apedawi-cs/Calibrated-uncertainty/blob/master/log_density.png">
