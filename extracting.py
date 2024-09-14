import SimpleITK as sitk
from radiomics import featureextractor, getTestCase

file_path = 'E:\\dataset\\archive2021\\BraTS2021_Training_Data'
img_name = '00000'
path_to_img = file_path + ("\\BraTS2021_" + img_name) + "\\BraTS2021_" + img_name + "_flair.nii"
path_to_mask =file_path + ("\\BraTS2021_" + img_name) + "\\BraTS2021_" + img_name + "_seg.nii"
imageName, maskName = path_to_img, path_to_mask

print('imageName, maskName', imageName, maskName)
if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
    print('Error getting testcase!')
# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline
#Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
