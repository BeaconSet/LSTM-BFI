# LSTM-BFI
Blood velocity predictor based on g2tau figures.

*All credit to Siwei Liu from Meiji University.*

# Introduction
Diffuse correlation spectroscopy (DCS) is a noninvasive technique that derives blood flow information from measurements of the temporal intensity fluctuations of multiply scattered light. Blood flow index (BFI) and especially its variation was demonstrated to be approximately proportional to absolute blood flow. 
**This learning model is built on a LSTM architecture to predict BFI from DCS data.**
<p align="center">
  <img src="/figure/g2tau1.JPG/">
</p>
<p align="center" href="">
  Fig 1. g2tau curves, all 3 trails were under the same blood velocity circumstance and recorded simultaneously.
</p>
We imported 691 pieces of DCS g2tau time-series data, each contains 3 trials (1cm, 2cm, 3cm DCS) and being fully labeled. These phantom data will be fed into the LSTM model and BFI prediction will be produced. Read the main.py for further information.


# Installation
Must be running on Python 3.8 or above.
To install, fork or clone the repository and go to the downloaded directory,
then run

```
pip install -r requirements.txt
```

### Requirements we use

mne
numpy
scipy
scikit-learn
matplotlib
pandas


# For HSME Lab Fellows
健康医工学研究室のみなさん，ここまでご覧いただき感謝申し上げます。

If you want to use this model for other proposes, ***please make sure your dataset is in .csv structure instead of a .mat file.*** 
<p align="center">
  <img src="/g2tau.PNG/">
</p>

***For privacy reasons, dataset has been moved to our lab google drive, please download from　M2>劉>DCSデータ.***
