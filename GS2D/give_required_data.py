import cv2
import torch
import numpy as np


def interpolate_colors(image_array, coord, w, h, device):
    coord_floor = coord.astype(int)
    coord_frac = coord - coord_floor
    coord_ceil = np.clip(coord_floor + 1, a_min=[0, 0], a_max=[w - 1, h - 1])

    color_00 = image_array[coord_floor[:, 1], coord_floor[:, 0]]
    color_11 = image_array[coord_ceil[:, 1], coord_ceil[:, 0]]
    color_01 = image_array[coord_floor[:, 1], coord_ceil[:, 0]]
    color_10 = image_array[coord_ceil[:, 1], coord_floor[:, 0]]

    dx = np.expand_dims(coord_frac[:, 0], axis=1)
    dy = np.expand_dims(coord_frac[:, 1], axis=1)

    colour_values = (
            (1 - dx) * (1 - dy) * color_00 +
            dx * (1 - dy) * color_10 +
            (1 - dx) * dy * color_01 +
            dx * dy * color_11
    )

    colour_values = torch.tensor(colour_values, device=device).float()
    return colour_values


def coords_normalize(coords, image_size):
    # normalizing pixel coordinates [-1,1], cood:(w,h)
    coords = (coords / [image_size[1], image_size[0]])
    center_coords_normalized = np.array([0.5, 0.5])
    coords = (center_coords_normalized - coords) * 2.0

    return coords


def get_colour(image_array, coord, image_size, device):    # image(h,w), coord(w,h)
    assert (image_array.shape[0] == image_size[0]) & (image_array.shape[1] == image_size[1]), f"The size of input image_array({image_array.shape[:2]}) must match the image_size({image_size})!"
    h, w = image_array.shape[:2]

    # reverse coords
    input_coord = (np.array([0.5, 0.5]) - coord / 2.0)
    input_coord = input_coord * [image_size[1],image_size[0]]

    return interpolate_colors(image_array, input_coord, w, h, device)


def give_required_data(input_coords, image_size, image_array, device):
    coords = torch.tensor(input_coords / [image_size[1], image_size[0]], device=device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords = (center_coords_normalized - coords) * 2.0

    colour_values_tensor = interpolate_colors(image_array, input_coords, image_size[1], image_size[0])

    return colour_values_tensor, coords


def coords_reverse(coords, image_size, device):
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    input_coords = (center_coords_normalized - coords / 2.0).clone().detach().to(device).float()
    input_coords = torch.mul(input_coords, torch.tensor([image_size[1], image_size[0]], device=device).float())

    return input_coords