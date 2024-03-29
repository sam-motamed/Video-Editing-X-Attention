#This file is the modified version of the same file name from: https://github.com/Sainzerjj/Free-Guidance-Diffusion/blob/master/free_guidance.py

import torch
from torch import tensor
from einops import rearrange
import numpy as np
import fastcore.all as fc
import math
import torch.nn.functional as F
from sklearn.decomposition import PCA
import torchvision.transforms as T
from PIL import Image
import os
from scipy.ndimage import label, generate_binary_structure
from copy import deepcopy
import copy
from functools import partial
# the calculation of character or attribute
import matplotlib.pyplot as plt


def keep_top_percent(tensor, percent=1):
    # Flatten the tensor
    flattened_tensor = tensor.flatten()
    
    # Sort the flattened tensor in descending order
    sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)
    
    # Determine the threshold value for the top percentile
    percentile_index = int(len(sorted_tensor) * (percent / 100))
    threshold_value = sorted_tensor[percentile_index]
    
    # Reshape the threshold value to the shape of the original tensor
    threshold_tensor = threshold_value.expand_as(tensor)
    
    # Create a mask to zero out values below the threshold
    mask = tensor >= threshold_tensor
    
    # Apply the mask
    result_tensor = tensor * mask
    
    return result_tensor



def draw_circular_target_attention(tensor, height, width):
    # Calculate the centroid and radius of the circle
    centroid = ( 4.5* height) // 9
    center_x =  ( 4.5 * width) // 9
    radius = height // 18

    # Create a grid of coordinates
    x_grid, y_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    # Calculate distances from each pixel to the centroid
    distances = torch.sqrt((x_grid - centroid)**2 + (y_grid - center_x)**2)
    
    # Create a mask for pixels within the circle
    circle_mask = distances <= radius
    
    # Set values inside the circle to 1 for all channels
    for i in range(tensor.size(0)):
        tensor[i, ~circle_mask.flatten()] = 0
        tensor[i, circle_mask.flatten()] = 1
    return tensor

def draw_rectangular_target_attention(tensor, height, width):
    start_x = int(2* width // 7)
    start_y = int(1 * height // 7)
    rect_height = 3 * int(height // 5)
    rect_width = 2 * int(width // 6)
    # Extract the number of channels
    num_channels = tensor.size(0)
    
    # Create a mask tensor of the same shape as the input tensor
    mask = torch.zeros_like(tensor.view(num_channels, height, width))
    
    # Determine the end coordinates of the rectangle
    end_x = min(start_x + rect_height, height)
    end_y = min(start_y + rect_width, width)
    
    # Iterate over each channel
    for c in range(num_channels):
        # Fill the rectangle area in the mask with 1s for each channel
        mask[c, start_x:end_x, start_y:end_y] = 1

    # Apply the mask to the input tensor
    tensor = tensor.view(num_channels, height, width) * (1 - mask)  # Set outside values to 0
    
    tensor += mask  # Set inner rectangle values to 1
    
    return tensor.view(num_channels, -1)


    return tensor


def gaussian_kernel(kernel_size, sigma):
    """
    Create a 2D Gaussian kernel
    """
    kernel = torch.tensor([[np.exp(-(x**2 + y**2) / (2. * sigma**2)) for x in range(-kernel_size//2 + 1, kernel_size//2 + 1)] for y in range(-kernel_size//2 + 1, kernel_size//2 + 1)])
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Apply Gaussian smoothing to a 2D tensor
    """
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    # Add dimensions to the kernel for compatibility with conv2d function
    kernel = kernel.unsqueeze(0).unsqueeze(0).cuda()
    # Convert input tensor to float
    input_tensor = input_tensor.to(torch.float64).cuda()
    # Apply convolution
    smoothed = F.conv2d(input_tensor.unsqueeze(0), kernel, padding=kernel_size//2)
    return smoothed.squeeze(0)




def normalize(x): 
    return (x - x.min()) / (x.max() - x.min())

def filter_tensor_by_value(tensor):
    value = torch.max(tensor)
    filtered_tensor = torch.zeros_like(tensor)
    filtered_tensor[tensor == value] = value
    return filtered_tensor

def threshold_attention(attn, s=10):
    norm_attn = s * (normalize(attn) - 0.5)
    #print(normalize(norm_attn.sigmoid()))
    return normalize(norm_attn.sigmoid())

def get_shape(attn, s=20): 
    return threshold_attention(attn, s)

def get_size(attn): 
    return 1/attn.shape[-1] * threshold_attention(attn).sum((0,1)).mean()

def get_c(cross_attention_tensor):
 
    cross_attention_tensor = cross_attention_tensor
    x_coords, y_coords = torch.meshgrid(torch.arange(cross_attention_tensor.shape[0]),torch.arange(cross_attention_tensor.shape[1]))

    # Compute the weighted sum of x and y coordinates
    weighted_x = (x_coords.cuda() * cross_attention_tensor.cuda()).sum()
    weighted_y = (y_coords.cuda() * cross_attention_tensor.cuda()).sum()

    # Compute the total weight
    total_weight = cross_attention_tensor.sum().cuda()

    # Compute the centroid coordinates
    centroid_x = weighted_x / total_weight
    centroid_y = weighted_y / total_weight
    #print(torch.stack([centroid_x, centroid_y]))
    return torch.stack([centroid_x, centroid_y])
    
    #return centroid_x.cpu().item(), .cpu().item()

def get_centroid(attn):
    centeroid_per_frame_window = []
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    hs = torch.arange(h).view(-1, 1, 1).to(attn.device)
    ws = torch.arange(w).view(1, -1, 1).to(attn.device)
    attn = rearrange(attn.mean(0), '(h w) d -> h w d', h=h)
    weighted_w = torch.sum(ws * attn, dim=[0,1])
    weighted_h = torch.sum(hs * attn, dim=[0,1])
    return torch.stack([weighted_w, weighted_h]) / attn.sum((0,1))


def get_appearance(attn, feats):
    # attn_fit = attn.permute(1, 0)
    # attn_fit = attn_fit.detach().cpu().numpy()
    # pca = PCA(n_components=3)
    # pca.fit(attn_fit)
    # feature_maps_pca = pca.transform(attn_fit)  # N X 3
    # pca_img = feature_maps_pca.reshape(1, -1, 3)  # B x (H * W) x 3
    # pca_img = pca_img.reshape(32, 32, 3)
    # pca_img_min = pca_img.min(axis=(0, 1))
    # pca_img_max = pca_img.max(axis=(0, 1))
    # pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    # pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    # pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
    # pca_img.save(os.path.join(f"1.png"))
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    shape = get_shape(attn).detach().mean(0).view(h,w,attn.shape[-1])
    feats = feats.mean((0,1))[:,:,None]
    return (shape * feats).sum() / shape.sum()

def get_attns(attn_storage):
   
    if attn_storage is not None:
        origs = attn_storage.maps('ori')
        #if "w" in origs.keys():
        #del(origs["w"])

        edits = attn_storage.maps('edit')
        #if "w" in edits.keys():
        #del(edits["w"])
    return origs, edits


def E(orig_attns, edit_attns, indices, tau, frame):
    print(indices)
    print("frame", frame)
    shapes = []
    num_frames = 16
    
    for f in range(num_frames):
        deltas = []
        delta = torch.tensor(0).to(torch.float16).cuda()
        out = []
        i = 0
        
        for location in ["down","mid", "up"]:
            for o in [1,2]:

                

                for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                    edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                    ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                    window_size = int(ori_attn_map.shape[0] // num_frames)
                    #print(ori_attn_map[f * window_size:f * window_size + window_size,:,o].shape)
                    orig, edit = ori_attn_map[f * window_size:f * window_size + window_size,:,o], edit_attn_map[f * window_size:f * window_size + window_size,:,o]
                    h = w = int(tensor(orig.shape[1]).sqrt().item())
                    orig_copy = copy.deepcopy(orig)
                    #ori = draw_filled_rectangle_per_channel(orig, h, w)
                    ori = draw_circular_target_attention(orig, h, w)
                    #ori = orig
                    #print(ori.shape, orig.shape)
                    if h in [16, 8]:
                        i += 1
                        
                        if len(ori.shape) < 3: ori, edit = ori[...,None], edit[...,None]
                        mean_image = torch.sum(orig_copy, axis=0).view(h, w)
                        mean_image_np = mean_image.detach().cpu().numpy()
                        #print(get_shape(orig).shape, orig.shape)
                        tau_mean_image = torch.sum(roll_shape(get_shape(ori), 'up', 0.02 * f), dim=0).view(h, w)
                        out.append(tau_mean_image)
                        tau_mean_image_np = tau_mean_image.detach().cpu().numpy()

                        # Plot the image
                        '''print(h * w)
                        plt.imshow(mean_image_np, cmap='gray')  # You can choose any colormap you prefer
                        plt.colorbar()
                        plt.title("being edited")
                        plt.show()'''

                        '''plt.imshow(tau_mean_image_np, cmap='gray')  # You can choose any colormap you prefer
                        plt.colorbar()
                        plt.title("target")
                        plt.show()'''


                        test = torch.sum(ori, dim=0).unsqueeze(0)
                        
                        #control how the target attention moves at each frame
                        first_roll = roll_shape(get_shape(ori), 'right', 0.00 * f)
                        delta = (get_shape(roll_shape(first_roll, 'down', 0.0 * f)) - get_shape(edit)).pow(2).mean()
                        
                
                    else:

                        delta = torch.tensor(0).to(torch.float16).cuda()
                    deltas.append(delta)
        print(i)
        shapes.append(torch.stack(deltas).mean())
    return torch.stack(shapes).sum()






def fix_appearances(orig_attns, ori_feats, edit_attns, edit_feats, indices):
    appearances = []
    sh = list(ori_feats.size())

    num_frames = 16
    for o in indices[:10]:
        for f in range(num_frames):
            
            window_size = int(edit_attns['up'][-1].shape[0] // num_frames)
            orig = torch.stack([a[:,:,o] for a in edit_attns['up'][-1:]]).mean(0)
            h = w = int(tensor(orig.shape[1]).sqrt().item())
            #edit = torch.stack([b[:,:,o] for b in orig_attns['up'][-3:]]).mean(0)
            orig = torch.stack([a[f * window_size:f * window_size + window_size,:,o] for a in edit_attns['up'][-1:]]).mean(0)
            edit = torch.stack([b[f * window_size:f * window_size + window_size,:,o] for b in orig_attns['up'][-1:]]).mean(0)
            edit_obj = torch.stack([b[f * window_size:f * window_size + window_size,:,2] for b in orig_attns['up'][-1:]]).mean(0)
            orig_obj = torch.stack([a[f * window_size:f * window_size + window_size,:,o] for a in edit_attns['up'][-1:]]).mean(0)
            ori = draw_circle2(orig_obj, h, w)
            if o == 2:
                #edit = draw_circle(edit_obj, h, w)
                #edit = roll_shape(edit, 'left', 0.05 * f)
                #edit = roll_shape(edit, 'up', 0.05 * f)
                #orig = ori * orig
                appearances.append((get_appearance(orig,ori_feats[f].view(1, sh[1], sh[2], sh[3])) - get_appearance(edit,edit_feats[f].view(1, sh[1], sh[2], sh[3]))).pow(2).mean())

            else:
                '''edi = draw_circle(edit_obj, h, w)
                edi.sub(1)
                edi * -1'''

                '''ori.sub(1)
                ori * -1'
                orig = ori * orig'''


                #print(edi.shape, edit.shape)
                #edit = edi *  edit
                print(edit.shape)
                appearances.append((get_appearance(orig,ori_feats[f].view(1, sh[1], sh[2], sh[3])) - get_appearance(edit,edit_feats[f].view(1, sh[1], sh[2], sh[3]))).pow(2).mean() * 2)
    return torch.stack(appearances).sum()








def fix_sizes(orig_attns, edit_attns, indices, tau=0.5):
    sizes = []
    num_frames = 18
    for location in ["mid", "down", "up"]:
        for o in indices[:15]:
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                for f in range(num_frames):
                    edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                    ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                    window_size = int(ori_attn_map.shape[0] // num_frames)
                    orig, edit = ori_attn_map[f * window_size:f * window_size + window_size,:,o], edit_attn_map[f * window_size:f * window_size + window_size,:,o]
                    #orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                    sizes.append((tau*get_shape(orig) - get_shape(edit)).pow(2).mean())
                    #print(tau*get_shape(orig), get_shape(edit)).pow(2).mean()
    return torch.stack(sizes).mean()








def position_deltas(orig_attns, edit_attns, indices, target_centroid=None):
    positions = []
    for location in ["mid", "down", "up"]:
        for o in indices:
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                target = tensor(target_centroid) if target_centroid is not None else get_centroid(orig)
                positions.append(target.to(orig.device) - get_centroid(edit))
    return torch.stack(positions).mean()

def roll_shape(x, direction='up', factor=0.5):
    h = w = int(math.sqrt(x.shape[-2]))
    mag = (0,0)
    if direction == 'up': mag = (int(-h*factor),0)
    elif direction == 'down': mag = (int(h*factor),0)
    elif direction == 'right': mag = (0,int(w*factor))
    elif direction == 'left': mag = (0,int(-w*factor))
    shape = (x.shape[0], h, h, x.shape[-1])
    x = x.view(shape)
    move = x.roll(mag, dims=(1,2))
    return move.view(x.shape[0], h*h, x.shape[-1])

def enlarge(x, scale_factor=1):
    assert scale_factor >= 1
    h = w = int(math.sqrt(x.shape[-2]))
    x = rearrange(x, 'n (h w) d -> n d h w', h=h)
    x = F.interpolate(x, scale_factor=scale_factor)
    new_h = new_w = x.shape[-1]
    x_l, x_r = (new_w//2) - w//2, (new_w//2) + w//2
    x_t, x_b = (new_h//2) - h//2, (new_h//2) + h//2
    x = x[:,:,x_t:x_b,x_l:x_r]
    return rearrange(x, 'n d h w -> n (h w) d', h=h) * scale_factor

def shrink(x, scale_factor=1):
    assert scale_factor <= 1
    h = w = int(math.sqrt(x.shape[-2]))
    x = rearrange(x, 'n (h w) d -> n d h w', h=h)
    sf = int(1/scale_factor)
    new_h, new_w = h*sf, w*sf
    x1 = torch.zeros(x.shape[0], x.shape[1], new_h, new_w).to(x.device)
    x_l, x_r = (new_w//2) - w//2, (new_w//2) + w//2
    x_t, x_b = (new_h//2) - h//2, (new_h//2) + h//2
    x1[:,:,x_t:x_b,x_l:x_r] = x
    shrink = F.interpolate(x1, scale_factor=scale_factor)
    return rearrange(shrink, 'n d h w -> n (h w) d', h=h) * scale_factor

def resize(x, scale_factor=1):
    if scale_factor > 1: return enlarge(x)
    elif scale_factor < 1: return shrink(x)
    else: return x




# guidance functions
def edit_layout(attn_storage, indices, appearance_weight=0.5, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    return appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices, **kwargs)

def edit_appearance(attn_storage, indices, shape_weight=1, **kwargs):
    origs, edits = get_attns(attn_storage)
   
    return shape_weight * fix_shapes_l1(origs, edits, indices)

def resize_object_by_size(attn_storage, indices, relative_size=4, shape_weight=1, size_weight=1, appearance_weight=0.1, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, indices)
    appearance_term = appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices)
    size_term = size_weight * fix_sizes(origs, edits, obj_idx, tau=relative_size)
    return shape_term + appearance_term + size_term

def resize_object_by_shape(attn_storage, indices, tau=fc.noop, shape_weight=0.5, size_weight=10, appearance_weight=0.5, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    # orig_selfs = [v['orig'] for k,v in attn_storage.storage.items() if 'attn1' in k][-1]
    # edit_selfs = [v['edit'] for k,v in attn_storage.storage.items() if 'attn1' in k][-1]
    if len(indices) > 1:
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, other_idx)
    appearance_term = appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices)
    size_term = size_weight * fix_shapes_l3(origs, edits, obj_idx, tau=tau)
    # self_term = self_weight*fix_selfs(orig_selfs, edit_selfs)
    return shape_term  + size_term + appearance_term

def move_object_by_centroid(attn_storage, indices, target_centroid=None, shape_weight=1, size_weight=1, appearance_weight=0.5, position_weight=1, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, indices)
    appearance_term = appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices)
    size_term = size_weight * fix_sizes(origs, edits, obj_idx)
    position_term = position_weight * position_deltas(origs, edits, obj_idx, target_centroid=target_centroid)
    return shape_term + appearance_term + size_term + position_term

def edit_by_E(attn_storage, indices, tau=fc.noop, shape_weight=1, appearance_weight=1, position_weight=8, ori_feats=None, edit_feats=None, frame=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])

    move_term = position_weight * E(origs, edits, obj_idx, tau=tau, frame=frame) 
    #print(move_term  , shape_term , appearance_term)
    return move_term  

def fix_appearances_by_feature(ori_feats, edit_feats, indices):
    appearances = []
    for o in indices[:10]:
        appearances.append((ori_feats - edit_feats).pow(2).mean())
    return torch.stack(appearances).mean()

def edit_layout_by_feature(attn_storage, indices, appearance_weight=0.5, ori_feats=None, edit_feats=None, **kwargs):
    return appearance_weight * fix_appearances_by_feature(ori_feats, edit_feats, indices)