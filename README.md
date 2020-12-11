# Bicycle-GAN

## Dependency

* python 3.8
* pytorch 1.7.0
* ..(see in requirements.txt)

## File Structure

* src
  - datasets.py
  - inference.py
  - models.py
  - train.py

* data
  - train
  - val

* checkpoint(this can be automatically generated after running train.py)
  - bicyclegan_0_999.pt
  - bicyclegan_1_1999.pt
  - ...
  - bicyclegan_18_18999.pt
  - bicyclegan_19_19999.pt

## Usage

train
```
python train.py
```
inference
```
python inference.py
```
