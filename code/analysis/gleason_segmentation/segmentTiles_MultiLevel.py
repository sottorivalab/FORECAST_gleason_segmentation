# Produces patchwise whole slide image segmetation for a multi-level UNet model

import itertools
import os
import glob
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py as h5
from ResNetUNet import ResNetUNet
from ResNetUNetEnsemble import ResNetUNetEnsemble
from torchvision import transforms

def segmentTiles_MultiLevel(tile_path, classifier_path, sub_result_path, out_path, image_size, segmentation_path=None, tile_size=[2000, 2000], block_size=[500, 500], block_offset=[250, 250], block_in_size=[224, 224], block_levels=[0, 1, 2], out_level=None, label_colours=[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.75, 0.75, 0.75]], ignore_border=10, batch_size=30, gpu=True, overwrite=False, norm=None):
    def read_image_region(image_region, image_size, tile_size, tile_path, ext='jpg'):
        tile_grid_size = [math.ceil(image_size[0]/tile_size[0]), math.ceil(image_size[1]/tile_size[1])]
        
        tile_range = [[math.floor(image_region[0][0]/tile_size[0]), math.ceil(image_region[0][1]/tile_size[0])], [math.floor(image_region[1][0]/tile_size[1]), math.ceil(image_region[1][1]/tile_size[1])]]
        
        image = ()
        
        for x in range(tile_range[0][0], tile_range[0][1]):
            image_row = ()
            
            for y in range(tile_range[1][0], tile_range[1][1]):
                tile_region = [[max(image_region[0][0]-(x*tile_size[0]), 0), min(image_region[0][1]-(x*tile_size[0]), tile_size[0])], [max(image_region[1][0]-(y*tile_size[1]), 0), min(image_region[1][1]-(y*tile_size[1]), tile_size[1])]]
                
                if x < 0 or x >= tile_grid_size[0] or y < 0 or y >= tile_grid_size[1]:
                    tile = np.zeros((tile_region[1][1] - tile_region[1][0], tile_region[0][1] - tile_region[0][0], 3)).astype(np.uint8)
                else:
                    tile_idx = x + (y*tile_grid_size[0])
                    tile_image_path = os.path.join(tile_path, 'Da'+str(tile_idx)+'.'+ext)

                    if os.path.exists(tile_image_path):
                        tile = cv2.imread(tile_image_path)

                        padding = [max(tile_region[0][1] - tile.shape[1], 0), max(tile_region[1][1] - tile.shape[0], 0)]
                        tile = tile[tile_region[1][0]:(tile_region[1][1]-padding[1]), tile_region[0][0]:(tile_region[0][1]-padding[0]), ::-1]
                        tile = np.pad(tile, ((0, padding[1]), (0, padding[0]), (0, 0)), mode='constant', constant_values=0)
                    else:
                        tile = np.zeros((tile_region[1][1] - tile_region[1][0], tile_region[0][1] - tile_region[0][0], 3)).astype(np.uint8)

                image_row = image_row + (tile, )
            
            image = image + (np.concatenate(image_row, axis=0), )
            
        image = np.concatenate(image, axis=1)
        
        return image
        
    def write_sub_results(sub_results, image_region, image_size, tile_size, n_labels, sub_result_path, chunk_size):
        tile_grid_size = [math.ceil(image_size[0]/tile_size[0]), math.ceil(image_size[1]/tile_size[1])]
        
        tile_range = [[max(math.floor(image_region[0][0]/tile_size[0]), 0), min(math.ceil(image_region[0][1]/tile_size[0]), tile_grid_size[0])], [max(math.floor(image_region[1][0]/tile_size[1]), 0), min(math.ceil(image_region[1][1]/tile_size[1]), tile_grid_size[1])]]

        for x in range(tile_range[0][0], tile_range[0][1]):           
            for y in range(tile_range[1][0], tile_range[1][1]):
                tile_region = [[max(image_region[0][0]-(x*tile_size[0]), 0), min(image_region[0][1]-(x*tile_size[0]), tile_size[0])], [max(image_region[1][0]-(y*tile_size[1]), 0), min(image_region[1][1]-(y*tile_size[1]), tile_size[1])]]
                result_region = [[tile_region[0][0]+(x*tile_size[0])-image_region[0][0], tile_region[0][1]+(x*tile_size[0])-image_region[0][0]], [tile_region[1][0]+(y*tile_size[1])-image_region[1][0], tile_region[1][1]+(y*tile_size[1])-image_region[1][0]]]

                tile_idx = x + (y*tile_grid_size[0])
                sub_result_file = os.path.join(sub_result_path, 'Da'+str(tile_idx)+'.h5')
                
                with h5.File(sub_result_file, 'a') as data:
                    if 'probability_sum' not in data.keys():
                        data.create_dataset('probability_sum', tuple(tile_size)+(n_labels, ), dtype='f', chunks=tuple(chunk_size)+(n_labels, ), compression='gzip')
                        data.create_dataset('probability_count', tuple(tile_size), dtype='i', chunks=tuple(chunk_size), compression='gzip')
                        
                    data['probability_sum'][tile_region[1][0]:tile_region[1][1], tile_region[0][0]:tile_region[0][1], :] += sub_results[result_region[1][0]:result_region[1][1], result_region[0][0]:result_region[0][1], :]
                    data['probability_count'][tile_region[1][0]:tile_region[1][1], tile_region[0][0]:tile_region[0][1]] += 1
                    
    def write_tiles(sub_result_path, out_path, segmentation_path, image_size, tile_size, label_colours):
        tile_grid_size = [math.ceil(image_size[0]/tile_size[0]), math.ceil(image_size[1]/tile_size[1])]
        
        for x in range(tile_grid_size[0]):           
            for y in range(tile_grid_size[1]):
                tile_idx = x + (y*tile_grid_size[0])
                
                out_tile_file = os.path.join(out_path, 'Da'+str(tile_idx)+'.png')
                
                if os.path.exists(out_tile_file):
                    print("Tile already exists for Da"+str(tile_idx)+'.png, skipping')
                else:
                    print("Writing tile: Da"+str(tile_idx)+'.png')
                    sub_result_file = os.path.join(sub_result_path, 'Da'+str(tile_idx)+'.h5')
                    
                    if segmentation_path is not None:
                        seg_tile_path = os.path.join(segmentation_path, 'Da'+str(tile_idx)+'.png')
                        
                        if os.path.exists(seg_tile_path):
                            seg_tile = cv2.imread(seg_tile_path, cv2.IMREAD_GRAYSCALE)
                            
                            #Workaround for when segmentation tile's size doesn't match image's size
                            #Can happen due to a bug in old CZI tiler 
                            out_tile_size = (min(image_size[1] - tile_size[1]*y, tile_size[1], seg_tile.shape[0]), min(image_size[0] - tile_size[0]*x, tile_size[0], seg_tile.shape[1]))

                            if seg_tile.shape[0] > out_tile_size[0]:
                                seg_tile = seg_tile[:out_tile_size[0], :]
                                
                            if seg_tile.shape[1] > out_tile_size[1]:
                                seg_tile = seg_tile[:, :out_tile_size[1]]
                        else:
                            out_tile_size = (min(image_size[1] - tile_size[1]*y - 1, tile_size[1]), min(image_size[0] - tile_size[0]*x - 1, tile_size[0]))
                            seg_tile = np.zeros(out_tile_size)
                    else:
                        seg_tile = None
                    
                    if os.path.exists(sub_result_file):
                        with h5.File(sub_result_file, 'r') as data:
                            tile_probs = data['probability_sum'][:out_tile_size[0], :out_tile_size[1], :]
                            tile_pcount = np.tile(np.expand_dims(data['probability_count'][:out_tile_size[0], :out_tile_size[1]], axis=2), (1, 1, len(label_colours)))
                            tile_probs[tile_pcount>0] = np.true_divide(tile_probs[tile_pcount>0], tile_pcount[tile_pcount>0])
                            pixel_labels = np.argmax(tile_probs, axis=2)
                            if seg_tile is not None:
                                pixel_labels[seg_tile==0] = 0
                            colour_tile = np.reshape(label_colours[pixel_labels.flatten(), ::-1], pixel_labels.shape + (label_colours.shape[1], ))
                    else:
                        colour_tile = np.zeros(out_tile_size + (3, ))
                        colour_tile[:,:,0] = label_colours[0, 0]
                        colour_tile[:,:,1] = label_colours[0, 1]
                        colour_tile[:,:,2] = label_colours[0, 2]

                    cv2.imwrite(out_tile_file, 255.0*colour_tile)
 
    if segmentation_path is not None and len(segmentation_path) == 0:
        segmentation_path = None
        
    if out_level == None:
        out_level = block_levels[0]
        
    if norm is not None:
        reference_image = cv2.imread(norm)[:, :, ::-1].astype(np.float32)/255.0

    model = torch.load(classifier_path, map_location=lambda storage, loc: storage)
    
    if gpu:
        model = model.cuda().eval()
    else:
        model = model.cpu().eval()
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    label_colours = np.array(label_colours)
    
    os.makedirs(sub_result_path, exist_ok=True)

    block_out_size = [x*(2**block_levels[0]) for x in block_size]

    blocks = torch.tensor(np.zeros((batch_size, 3*len(block_levels), block_in_size[0], block_in_size[1])), dtype=torch.float)
    
    block_centres = list(itertools.product(range(math.floor(block_size[0]/2), image_size[0], block_offset[0]), range(math.floor(block_size[1]/2), image_size[1], block_offset[1])))
    
    if gpu:
        blocks = blocks.cuda()
    else:
        blocks = blocks.cpu()
    
    checkpoint_file = os.path.join(sub_result_path, 'checkpoint.h5')
    
    print("Checking for existing progress on "+os.path.basename(sub_result_path)+".")
    if os.path.exists(checkpoint_file):
        with h5.File(checkpoint_file, 'a') as checkpoint:
            if 'number_of_blocks' in checkpoint.keys() and 'last_block_processed' in checkpoint.keys() and checkpoint['number_of_blocks'][()] == len(block_centres):
                i = checkpoint['last_block_processed'][()]
                print("Checkpoint file found, continuing from block "+str(i)+".")
            else:
                #Something has changed with the configuration of the segmentation since the last saved run
                #To prevent conflicts, delete progress so far and start again.
                print("Checkpoint file found, but is inconsistent with current configuration.")
                print("Discarding current progress and continuing from beginning.")
                
                for sub_result_file in glob.glob(os.path.join(sub_result_path, 'Da*.h5')):
                    os.remove(sub_result_file)
                    
                i = 0
                
                if 'last_block_processed' in checkpoint.keys():
                    checkpoint['last_block_processed'][()] = i
                else:
                    checkpoint.create_dataset('last_block_processed', data=i)
                    
                if 'number_of_blocks' in checkpoint.keys():
                    checkpoint['number_of_blocks'][()] = len(block_centres)
                else:
                    checkpoint.create_dataset('number_of_blocks', data=len(block_centres))
                
    else:
        print("No checkpoint file found.")
        
        for sub_result_file in glob.glob(os.path.join(sub_result_path, 'Da*.h5')):
            os.remove(sub_result_file)
            
        i = 0
        
        with h5.File(checkpoint_file, 'w') as checkpoint:
            checkpoint.create_dataset('last_block_processed', data=i)
            checkpoint.create_dataset('number_of_blocks', data=len(block_centres))
    
    if i < len(block_centres):
        while i < len(block_centres):
            print("Processing block "+str(i+1)+" of "+str(len(block_centres)))
            
            j = 0
            
            batch_idxs = []
            
            while j < batch_size and i < len(block_centres):
                block_region = [[block_centres[i][0]-math.floor(block_size[0]*(2**out_level)/2), block_centres[i][0]+math.ceil(block_size[0]*(2**out_level)/2)], [block_centres[i][1]-math.floor(block_size[1]*(2**out_level)/2), block_centres[i][1]+math.ceil(block_size[1]*(2**out_level)/2)]]
                
                if segmentation_path is not None:
                    seg_tile = read_image_region(block_region, image_size, tile_size, segmentation_path, ext='png')
                else:
                    seg_tile = None
                    
                if seg_tile is None or np.any(seg_tile):
                    batch_idxs += [i]
                    
                    for l in range(0, len(block_levels)):
                        block_region = [[block_centres[i][0]-math.floor(block_size[0]*(2**block_levels[l])/2), block_centres[i][0]+math.ceil(block_size[0]*(2**block_levels[l])/2)], [block_centres[i][1]-math.floor(block_size[1]*(2**block_levels[l])/2), block_centres[i][1]+math.ceil(block_size[1]*(2**block_levels[l])/2)]]
                        block_level = read_image_region(block_region, image_size, tile_size, tile_path, ext='jpg')
                        block_level = cv2.resize(block_level, tuple(block_in_size), interpolation=cv2.INTER_AREA)
                        if norm is not None:
                            block_level = norm_reinhard(block_level.astype(np.float32)/255.0, reference_image)
                            block_level[block_level<0.0] = 0.0
                            block_level[block_level>1.0] = 1.0
                        blocks[j, 3*l:3*(l+1), :, :] = transform(block_level)
                    
                    j+=1
                i+=1
            
            output = F.softmax(model(blocks), dim=1).data
            
            if gpu:
                output = output.cpu()
                
            output = output.numpy().transpose([0, 2, 3, 1])
                
            for k in range(0, j):
                block_region = [[block_centres[batch_idxs[k]][0]-math.floor(block_size[0]*(2**block_levels[0])/2)+ignore_border, block_centres[batch_idxs[k]][0]+math.ceil(block_size[0]*(2**block_levels[0])/2)-ignore_border], [block_centres[batch_idxs[k]][1]-math.floor(block_size[1]*(2**block_levels[0])/2)+ignore_border, block_centres[batch_idxs[k]][1]+math.ceil(block_size[1]*(2**block_levels[0])/2)-ignore_border]]
                block_mask = np.concatenate(tuple(np.expand_dims(cv2.resize(output[k, :, :, d], tuple(block_out_size), interpolation=cv2.INTER_AREA), axis=2) for d in range(output.shape[3])), axis=2)

                if ignore_border > 0:
                    block_mask = block_mask[ignore_border:-ignore_border, ignore_border:-ignore_border, :]
                    
                write_sub_results(block_mask, block_region, image_size, tile_size, len(label_colours), sub_result_path, block_offset)
                
                with h5.File(checkpoint_file, 'a') as checkpoint:
                    checkpoint['last_block_processed'][()] = batch_idxs[k]+1
        
        with h5.File(checkpoint_file, 'a') as checkpoint:
            checkpoint['last_block_processed'][()] = len(block_centres)
                
    os.makedirs(out_path, exist_ok=True)
    
    write_tiles(sub_result_path, out_path, segmentation_path, image_size, tile_size, label_colours)
    
def norm_reinhard(source_image, target_image):

    source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2Lab)

    ms = np.mean(source_lab, axis=(0, 1))
    stds = np.std(source_lab, axis=(0, 1))
    
    if np.any(stds == 0):
        print("Warning: Stain normalisation failed due to StDev == 0 in input colour channel, leaving block unnormalised.")
        return source_image

    target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2Lab)

    mt = np.mean(target_lab, axis=(0, 1))
    stdt = np.std(target_lab, axis=(0, 1))

    norm_lab = np.copy(source_lab)

    norm_lab[:,:,0] = ((norm_lab[:,:,0]-ms[0])*(stdt[0]/stds[0]))+mt[0]
    norm_lab[:,:,1] = ((norm_lab[:,:,1]-ms[1])*(stdt[1]/stds[1]))+mt[1]
    norm_lab[:,:,2] = ((norm_lab[:,:,2]-ms[2])*(stdt[2]/stds[2]))+mt[2]

    weight_mask = (source_lab[:,:,0] - 80) / 10
    weight_mask[weight_mask < 0] = 0
    weight_mask[weight_mask > 1] = 1
    weight_mask = np.tile(weight_mask[:, :, np.newaxis], (1, 1, 3))

    norm_image = cv2.cvtColor(norm_lab, cv2.COLOR_Lab2RGB)
    norm_image = (source_image*weight_mask + norm_image*(1-weight_mask))

    return norm_image
