# Calibrated uncertainty
Playing around with ideas from a few papers:

[1] [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/pdf/1612.01474.pdf)

[2] [Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors](https://arxiv.org/pdf/1807.09289v2.pdf)

To run, just use `python example.py`.

### Required Python dependencies:
```
matplotlib==1.3.1
numpy==1.14.5
tensorflow==1.10.1
tensorflow_probability
```

---------

### Results

Data generating process:

<img src="https://github.com/apedawi-cs/Calibrated-uncertainty/blob/master/dgp.png" width="300">

Actual log density:

<img src="https://github.com/apedawi-cs/Calibrated-uncertainty/blob/master/logdensity_actual.png">

Estimated log density:

<img src="https://github.com/apedawi-cs/Calibrated-uncertainty/blob/master/logdensity_estimated.png">
