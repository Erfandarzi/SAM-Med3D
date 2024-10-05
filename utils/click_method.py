import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_next_click3D_torch(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):#, desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        point = None
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # if no mask is given, random click a negative point
        if point is None: 
            point = torch.Tensor([np.random.randint(sz) for sz in fn_masks[i].size()]).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1,1,-1).to(pred_masks.device) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1).to(pred_masks.device) 

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    


import edt
def get_next_click3D_torch_ritm(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fp_mask_single = F.pad(fp_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = fn_max_dist > fp_max_dist # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0) # use a erosion area
    to_point_mask = to_point_mask[None, None]
    # import pdb; pdb.set_trace()

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    



def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):

        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        # import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    

def get_uncertainty_map(prev_seg):
    prob_maps = torch.sigmoid(prev_seg)
    entropy = -(prob_maps * torch.log(prob_maps + 1e-10) + (1 - prob_maps) * torch.log(1 - prob_maps + 1e-10))
    return entropy

def get_next_click3D_torch_adaptive(prev_seg, gt_semantic_seg, num_points=1):
    print(f"Input shapes: prev_seg {prev_seg.shape}, gt_semantic_seg {gt_semantic_seg.shape}")
    print(f"Device: {prev_seg.device}")
    
    batch_size = prev_seg.shape[0]
    device = prev_seg.device
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(batch_size):
        print(f"Processing batch item {i}")
        
        # Calculate uncertainty map
        prob = torch.sigmoid(prev_seg[i, 0])
        uncertainty = -(prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6))
        
        # Combine uncertainty with error regions
        fn_sampling_map = uncertainty * fn_masks[i, 0].float()
        fp_sampling_map = uncertainty * fp_masks[i, 0].float()

        # Flatten and normalize the sampling map
        sampling_map = fn_sampling_map + fp_sampling_map
        flat_map = sampling_map.view(-1)
        flat_map = flat_map / (flat_map.sum() + 1e-10)

        # Sample an index based on the normalized map
        index = torch.multinomial(flat_map, 1)

        # Convert flat index back to 3D coordinates
        z = index // (sampling_map.shape[1] * sampling_map.shape[2])
        y = (index % (sampling_map.shape[1] * sampling_map.shape[2])) // sampling_map.shape[2]
        x = index % sampling_map.shape[2]
        point = torch.stack([z, y, x]).reshape(1, 1, 3)

        is_positive = fn_masks[i, 0, point[0, 0, 0], point[0, 0, 1], point[0, 0, 2]].item()

        batch_points.append(point)
        batch_labels.append(torch.tensor([[int(is_positive)]], device=device))

        print(f"Batch item {i}: Selected point shape {point.shape}")

    print(f"Final shapes: batch_points {len(batch_points)}, batch_labels {len(batch_labels)}")
    return batch_points, batch_labels
def get_next_click3D_torch_with_dice(prev_seg, gt_semantic_seg):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        # mask_gt = mask_gt.astype(bool)
        mask_gt = (mask_gt > 0)
        
        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2*volume_intersect / volume_sum

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)


    for i in range(gt_semantic_seg.shape[0]):

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # bp = torch.tensor(point[1:]).reshape(1,1,3) 
        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    return batch_points, batch_labels, (sum(dice_list)/len(dice_list)).item()    


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_point(point, label, ax):
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='red'))
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='green'))
    # plt.scatter(point[0], point[1], label=label)



if __name__ == "__main__":
    gt2D = torch.randn((2,1,256, 256)).cuda()
    prev_masks = torch.zeros_like(gt2D).to(gt2D.device)
    batch_points, batch_labels = get_next_click3D_torch(prev_masks.to(gt2D.device), gt2D)
    print(batch_points)