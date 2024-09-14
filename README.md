## Segmentation-guided Multi-modal Brain Tumors Survival Prediction Model Using Pseudo-labelling Approach

We used the https://github.com/lescientifik/open_brats2020 project as a framework to write our code

#### Training

First change your data source folder by modifying values in `src/config.py`

```python
BRATS_TRAIN_FOLDERS = "your-Path_to/brats2020/MICCAI_BraTS_2020_Data_Training"
BRATS_VAL_FOLDER = "your-Path_to/brats2020/MICCAI_BraTS_2020_Data_Valdation"
BRATS_TEST_FOLDER = "your-Path_to/brats2020/MICCAI_BraTS_2020_Data_Testing"
```

If you prefer not to hardcode this value, you can set them as variable environments.

Then, start training:

```
python -m src.train --devices 0 --width 48 --arch EquiUnet
```

For more details on the available option:
```
python -m src.train -h
```
