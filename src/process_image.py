import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = '../data/input/'
OUTPUT_DIR = '../data/output/'
LEFT_EXT = 'l.jpg'
RIGHT_EXT = 'r.jpg'
IMAGE_NAME = ['corridor','triclopsi2']
#IMAGE_NAME = ['test']


def load_image(name):
    '''
    Use cv2 to read the left and right images from the directory.
    '''
    image_left_path = os.path.join(INPUT_DIR, name + LEFT_EXT)
    image_left = cv2.imread(image_left_path, cv2.IMREAD_GRAYSCALE)
    image_left = image_left.astype('float32')
        
    image_right_path = os.path.join(INPUT_DIR, name + RIGHT_EXT)
    image_right = cv2.imread(image_right_path, cv2.IMREAD_GRAYSCALE)
    image_right = image_right.astype('float32')
        
    height = image_left.shape[0]
    width = image_left.shape[1]
    name_height_width = (name, height, width)
        
    return image_left, image_right, name_height_width


def plot_image(image_left, image_right, name):
    '''
    Plot the left and right images.
    Save the plot as a PNG file in the directory ./output
    '''  
    plt.figure(1, figsize=(20, 20))
    
    plt.subplot(131)
    plt.title(name + ' (Left Image)')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.imshow(image_left, cmap='gray')
        
    plt.subplot(132)
    plt.title(name + ' (Right Image)')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.imshow(image_right, cmap='gray')
        
    plt.savefig(OUTPUT_DIR + name)         
    plt.show()

    
def plot_disparity_map(d_map, name, param, median_filter=False, gaussian_filter=False):
    '''
    Plot the disparity map.
    Save the map as a PNG file in the directory ./output
    '''
    param_list = param.split()
    d = param_list[0]
    d = d.replace('d=','d')
    d = d.replace(',','')
    w = param_list[1]
    w = w.replace('w=','w')
    filename_suffix = '_' + d + '_' + w

    plt.figure(1, figsize=(20, 20))
        
    plt.subplot(131)
    plt.title(name + ' (Disparity Map)' + ' (' + param + ')')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.imshow(d_map, cmap='gray')

    if median_filter:
        d_map_median = cv2.medianBlur(d_map, 5)
        plt.subplot(132)
        plt.title(name + ' (Disparity Map + Median Filter) '  + '(' + param + ')')
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.imshow(d_map_median, cmap='gray')
        
    if gaussian_filter:
        d_map_median = cv2.GaussianBlur(d_map, (5,5), 0)
        plt.subplot(132)
        plt.title(name + ' (Disparity Map + Gaussian Filter)'  + '(' + param + ')')
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.imshow(d_map_median, cmap='gray')
            
    plt.savefig(OUTPUT_DIR + name + '_disparity_map' + filename_suffix) 
    plt.show()
    
    
def compute_cost(image_left, image_right, name_height_width, max_disparity, window_size):
    '''
    Compute the matching cost between support windows from left and right images.
    '''
    height = name_height_width[1]
    width = name_height_width[2]
    w = window_size//2
    cost = np.zeros((height, width, max_disparity)).astype('float32')
     
    for y in range(w, height - w):
        for x in range(w, width - w):
            for v in range(-w, w + 1):
                for u in range(-w, w + 1):
                    pixel_value_left = image_left[y+v,x+u]
                    for d in range(max_disparity):
                        pixel_value_right = image_right[y+v,x+u-d]
                        cost[y,x,d] += np.abs(pixel_value_left - pixel_value_right)
    return cost       


def compute_disparity(image_left, image_right, name_height_width,\
                      max_disparity, window_size,\
                      param, median_filter=False, gaussian_filter=False):
    '''
    Compute the disparity map (based on minimum cost).
    Process the disparities along the sides of the map.
    '''
    name = name_height_width[0]
    height = name_height_width[1]
    width = name_height_width[2]
    w = window_size//2
    cost = compute_cost(image_left, image_right, name_height_width,\
                        max_disparity, window_size)
    d_map = np.argmin(cost, axis=2).astype('float32')
    
    # Process pixels along the four borders (and corners) of d_map
    for y in range(height):
        for x in range(width):
            if x < w:
                d_map[y,x] = d_map[y,w]
            if x >= width-w:
                d_map[y,x] = d_map[y,width-w-1]
            if y < w:
                d_map[y,x] = d_map[w,x]
            if y >= height-w:
                d_map[y,x] = d_map[height-w-1,x]
            if x < w and y < w:
                d_map[y,x] = d_map[w,w]
            if x >= width-w and y < w:
                d_map[y,x] = d_map[w,width-w-1]  
              
    plot_disparity_map(d_map, name, param, median_filter, gaussian_filter)
    
    
# Run for all images        
for name in IMAGE_NAME:
    image_left, image_right, name_height_width = load_image(name)
    plot_image(image_left, image_right, name)
    
    start = time.time()
    compute_disparity(image_left, image_right, name_height_width, max_disparity=15, window_size=8, param = 'd=15, w=8', median_filter=True)
    end = time.time()
    print(f'{name} (d=15, w=8): Time taken = {end-start} seconds')  

    start = time.time()
    compute_disparity(image_left, image_right, name_height_width, max_disparity=15, window_size=12, param = 'd=15, w=12', median_filter=True)
    end = time.time()
    print(f'{name} (d=15, w=12): Time taken = {end-start} seconds')  
    
    start = time.time()
    #compute_disparity(image_left, image_right, name_height_width, max_disparity=15, window_size=16, param = 'd=15, w=16', median_filter=False, gaussian_filter=True)
    compute_disparity(image_left, image_right, name_height_width, max_disparity=15, window_size=16, param = 'd=15, w=16', median_filter=True, gaussian_filter=False)
    end = time.time()
    print(f'{name} (d=15, w=16): Time taken = {end-start} seconds')
    
    start = time.time()
    compute_disparity(image_left, image_right, name_height_width, max_disparity=10, window_size=16, param = 'd=10, w=16', median_filter=True)
    end = time.time()
    print(f'{name} (d=10, w=16): Time taken = {end-start} seconds')
    
    start = time.time()
    compute_disparity(image_left, image_right, name_height_width, max_disparity=20, window_size=16, param = 'd=20, w=16', median_filter=True)
    end = time.time()
    print(f'{name} (d=20, w=16): Time taken = {end-start} seconds')    
    