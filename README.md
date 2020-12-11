# Bicycle-GAN

## Dependency

* python 3.8
* pytorch 1.7.0
* ..(see requirements.txt)

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
After running inference.py, a folder named inference will be generated. 
* inference
  - eval_checkpoints
  - fake_images
  - real_images
  - results

