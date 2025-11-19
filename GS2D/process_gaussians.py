import csv
import os
import glob

import h5py
import numpy as np
import torch

# gaussian_params path
folder_path = "/home/ubuntu/datasets/Counting/UCF-Train-Val-Test/train/gs_params"

file_list = glob.glob(os.path.join(folder_path, "*.h5"))


def d_shape_loss(scale, shape_threshold=1.5):
    shape_ratio = torch.max(scale[:,0] / (scale[:,1]+1e-10), scale[:,1] / (scale[:,0]+1e-10))
    idxx = np.where(shape_ratio > shape_threshold)[0]
    shape_loss,idx = torch.max(torch.clamp(shape_ratio - shape_threshold, min=0), dim=-1)
    # print(shape_loss)

    return shape_loss.mean(),idx,idxx


output_csv_path = os.path.join(folder_path, "aa.csv")
out_num=0
# Open a CSV file for writing the results
with open(output_csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File Name', 'Loss'])

    # Iterate through all .npy files in the folder
    for file_name in os.listdir(folder_path):

        if file_name.endswith(".h5") and "bk" not in file_name:
            file_path = os.path.join(folder_path, file_name)

            with h5py.File(file_path, 'r') as f:
                params = f['params'][:]

            if not len(params)>0:
                print(f'{file_name} has no keypoints!')
                continue
            gs_params = params[:, :5]
            scales = gs_params[:, 2:4]

            scales = torch.tensor(scales)
            shape_loss, idx, idxx = d_shape_loss(scales)

            if np.isnan(shape_loss) or shape_loss > 0.:
                print(f"{file_name} shape loss:{shape_loss.detach().numpy()}, idx:{idx}, coord:{gs_params[idx,:2]}")
                out_num = out_num + 1

                # backup the file
                bk_path = file_path.replace(".h5", "_bk.h5")
                os.rename(file_path, bk_path)
                print("********************")
                print(scales)
                short_axis = torch.min(scales[:, 0], scales[:, 1])

                new_params = params.copy()
                scales[idxx, 0] = short_axis[idxx]
                scales[idxx, 1] = short_axis[idxx]
                print("##################")
                print(scales)

                scales = scales.detach().numpy()
                new_params[:,2:4] = scales
                with h5py.File(os.path.join(file_path), 'w') as f:
                    f.create_dataset('params', data=new_params, compression='gzip')

            # Write the file name and loss to the CSV file
            csv_writer.writerow([file_name, shape_loss.detach().numpy()])  # Assuming shape_loss is a scalar

print("Loss results saved to:", output_csv_path)
print(f"Outbounded shape: {out_num}/{len(file_list)}")