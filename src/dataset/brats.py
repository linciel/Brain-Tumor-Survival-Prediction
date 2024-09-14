import pathlib

import SimpleITK as sitk
import numpy as np
from sklearn.model_selection import KFold
from torch import tensor
from torch.utils.data.dataset import Dataset
import pandas as pd
from config import BRATS_SUR_FOLDER
import torch
from config import get_brats_folder
from dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, debug=False, data_aug=False,
                 no_seg=False, normalisation="minmax", bin_num=6, only_sur=False):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.debug = debug
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        df = pd.read_csv(BRATS_SUR_FOLDER, encoding="utf-8")
        self.sur_infos = np.array(df)
        sur = self.sur_infos[:, 2:3]
        sur = np.delete(sur, np.where(sur == -1))
        sur = sur.astype(np.float)
        self.bins = make_cuts(sur, bin_num)

        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            index = np.where(self.sur_infos[:, 0:1] == patient_id)
            sur_info = [-1, -1, -1, -1, -1, -1, -1,0]
            if len(index[0]) > 0:
                sur_info = self.sur_infos[np.where(self.sur_infos[:, 0:1] == patient_id)[0], :][0, 1:]
            sur_info = np.asarray(sur_info).astype(np.float)
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            # pseudo = [2] if sur_info[1] > 0 else [0]
            # pseudo=np.asarray(pseudo).astype(np.float64)
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                # id,age,sur_days ,sex,race,type,alive
                age=sur_info[0], sur_days=sur_info[1], sex=sur_info[2],
                race=sur_info[3], type=sur_info[4], alive=sur_info[5], pseudo=sur_info[6],
                sur_label=get_cuts_label(sur_info[1], self.bins),
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            if only_sur:
                if patient['sur_days'] != -1:
                    self.datas.append(patient)
            else:
                self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not
                         in ["id", "seg", "age", "sur_days", "sex", "race", "type", "alive", "sur_label","pseudo"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    sur_days=_patient['sur_days'], sur_label=_patient['sur_label'],
                    age=_patient['age'], sex=_patient['sex'], pseudo=_patient['pseudo'],
                    race=_patient['race'], type=_patient['type'], alive=_patient['alive'],
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True
                    )

    @staticmethod
    def load_nii(path_folder):
        # print(path_folder)
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3

def get_datasets(seed, debug, no_seg=False, on="train", full=False,
                 fold_number=0, normalisation="minmax",val=False):
    base_folder = pathlib.Path(get_brats_folder(on)).resolve()
    val_folder = pathlib.Path(get_brats_folder("val")).resolve()
    # print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    val_patients_dir = sorted([x for x in val_folder.iterdir() if x.is_dir()])
    if full:
        train_dataset = Brats(patients_dir, training=True, debug=debug,
                              normalisation=normalisation)
        bench_dataset = Brats(patients_dir, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        if val:
            val_dataset = Brats(val_patients_dir, training=True, debug=debug,no_seg=False,
                              normalisation=normalisation)
        else:
            val_dataset = Brats(val_patients_dir, training=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset,val_dataset, train_dataset.bins
    if no_seg:
        return Brats(patients_dir, training=False, debug=debug,
                     no_seg=no_seg, normalisation=normalisation)
    kfold = KFold(10, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    print("first idx of train", train_idx[0])
    print("first idx of test", val_idx[0])
    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_idx]
    # return patients_dir
    train_dataset = Brats(train, training=True, debug=debug,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=True, debug=debug,
                        normalisation=normalisation)
    bench_dataset = Brats(val, training=False, benchmarking=True, debug=debug,
                          normalisation=normalisation)
    return train_dataset, bench_dataset, val_dataset, train_dataset.bins

def make_cuts(x, cuts, equidistant=False, min_=0, max_=2000):
    x = tensor(x)
    min_ = min(x)
    max_2 = max(x)
    if max_2 < max_:
        max_ = max_2
    if equidistant:
        cuts = torch.linspace(min_, max_, cuts)
    else:
        x = x.sort()[0][:-1]
        n = len(x)
        cuts -= 1
        idxs = tensor([n // cuts] * cuts)
        t = torch.zeros_like(idxs)
        t[:n - idxs.sum()] = 1
        idxs = (idxs + t).cumsum(0) - 1
        cuts = torch.cat([tensor([min_]), x[idxs]])
    cuts = cuts[1:] - cuts[:-1]
    return cuts

def get_cuts_label(x, cuts):
    if x == -1:
        return torch.FloatTensor([0.0] * len(cuts))
    i = 0
    label = []
    while i < len(cuts):
        if x >= cuts[i]:
            label.append(1.0)
            x = x - cuts[i]
        else:
            label.append(x / cuts[i])
            x = 0
        i += 1
    return torch.FloatTensor(label)
