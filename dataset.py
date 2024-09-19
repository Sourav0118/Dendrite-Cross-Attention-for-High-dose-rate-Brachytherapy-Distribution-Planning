import os
import torch
from torch.utils.data import Dataset
import nibabel as nib

class MedicalDataset(Dataset):
    def __init__(self, base_folder, transform=True):
        self.base_folder = base_folder
        self.subjects = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_folder = os.path.join(self.base_folder, self.subjects[idx])
        
        ct_path = os.path.join(subject_folder, 'CT.nii.gz')
        dose_path = os.path.join(subject_folder, 'dose.nii.gz')
        ctv_path = os.path.join(subject_folder, 'ctv.nii.gz')
        bladder_path = os.path.join(subject_folder, 'bladder.nii.gz')
        rectum_path = os.path.join(subject_folder, 'rectum.nii.gz')
        applicator_path = os.path.join(subject_folder, 'applicator.nii.gz')

        ct = self.load_nifti(ct_path)
        dose = self.load_nifti(dose_path)
        ctv = self.load_nifti(ctv_path)
        bladder = self.load_nifti(bladder_path)
        rectum = self.load_nifti(rectum_path)
        applicator = self.load_nifti(applicator_path)

        if self.transform:
            ct = self.ToTensor(ct)
            dose = self.ToTensor(dose)
            ctv = self.ToTensor(ctv)
            bladder = self.ToTensor(bladder)
            rectum = self.ToTensor(rectum)
            applicator = self.ToTensor(applicator)
            
            ct, dose, ctv, bladder, rectum, applicator = self.CentreCrop(ct, dose, ctv, bladder, rectum, applicator)

            ct = self.normalize(ct, 'std')
            dose = self.normalize(dose, 'scl')
                
        return {'ct': ct, 'ctv': ctv, 'applicator': applicator,
                'bladder': bladder, 'rectum': rectum, 'dose': dose}

    def load_nifti(self, file_path):
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return img_data

    def normalize(self, tensor, method):
        if method == 'std':
            means = torch.mean(tensor, dim=(0, 1), keepdim=True)
            stds = torch.std(tensor, dim=(0, 1), keepdim=True)
            stds[stds == 0] = 1e-10
            normalized_tensor = (tensor - means) / stds
        elif method == 'scl':
            maxs, mins = 50, 0
            diff = maxs - mins
            normalized_tensor = (tensor - mins) / diff
        else:
            raise ValueError("Normalization method not supported. Choose 'std' or 'scl'.")

        return normalized_tensor

    def ToTensor(self, data):
        data = torch.from_numpy(data).float()
        return data
   
    def CentreCrop(self, inputCT, dose, ctv, bladder, rectum, applicator):
        inputCT = inputCT.unsqueeze(0)
        ctv = ctv.unsqueeze(0)
        dose = dose.unsqueeze(0)
        bladder = bladder.unsqueeze(0)
        rectum = rectum.unsqueeze(0)
        applicator = applicator.unsqueeze(0)
        border = 10
        output_shape = (128, 128, 32)
        total_slices = inputCT.shape[-1]
        start_idx = (total_slices - 32) // 2
        end_idx = start_idx + 32
        # Find the bounding box of the mask
        indices = torch.nonzero(ctv)
        min_indices = indices.min(dim=0)[0][1:]
        max_indices = indices.max(dim=0)[0][1:]

        # Add border to the bounding box
        min_indices = torch.clamp(min_indices - border, min=0)
        max_indices = torch.clamp(max_indices + border, max=torch.tensor(inputCT.shape[1:]) - 1)

        # Calculate the center of the bounding box
        center = (min_indices + max_indices) // 2

        # Calculate the starting and ending indices for the crop
        start_indices = center - torch.tensor(output_shape) // 2
        end_indices = start_indices + torch.tensor(output_shape)

        # Ensure the indices are within bounds
        start_indices = torch.clamp(start_indices, min=0)
        end_indices = torch.clamp(end_indices, max=torch.tensor(inputCT.shape[1:]) - 1)

        # Crop the inputTensor
        cropped_ct = inputCT[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_idx:end_idx]
        cropped_dose = dose[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_idx:end_idx]
        cropped_ctv = ctv[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_idx:end_idx]
        cropped_bladder = bladder[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_idx:end_idx]
        cropped_rectum = rectum[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_idx:end_idx]
        cropped_applicator = applicator[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_idx:end_idx]
        # Calculate any necessary padding to achieve the desired output shape
        padding = []
        for i in range(3):
            pad_before = max(0, -start_indices[i].item())
            pad_after = max(0, end_indices[i].item() - inputCT.shape[i+1])
            padding.extend([pad_before, pad_after])

        # Apply padding if necessary
        if any(padding):
            cropped_ct = F.pad(cropped_ct, padding, mode='constant', value=0)
            cropped_dose = F.pad(cropped_dose, padding, mode='constant', value=0)
            cropped_ctv = F.pad(cropped_ctv, padding, mode='constant', value=0)
            cropped_bladder = F.pad(cropped_bladder, padding, mode='constant', value=0)
            cropped_rectum = F.pad(cropped_rectum, padding, mode='constant', value=0)
            cropped_applicator = F.pad(cropped_applicator, padding, mode='constant', value=0)

        # Center crop to the final output_shape (if necessary)
        final_shape = cropped_ct.shape[1:]
        crop_start = [(final_shape[i] - output_shape[i]) // 2 for i in range(3)]
        crop_end = [crop_start[i] + output_shape[i] for i in range(3)]

        final_cropped_ct = cropped_ct[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]
        final_cropped_dose = cropped_dose[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]
        final_cropped_ctv = cropped_ctv[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]
        final_cropped_bladder = cropped_bladder[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]
        final_cropped_rectum = cropped_rectum[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]
        final_cropped_applicator = cropped_applicator[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :]

        return final_cropped_ct, final_cropped_dose, final_cropped_ctv, final_cropped_bladder, final_cropped_rectum, final_cropped_applicator
