# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:15:11 2015

@author: Shihao Zhao
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
   
'''
Input: initial_image(reference image), x, y(coordinates), target_image
Output: The sum of 255 - (initial_image - target_image)^2
 ==> The larger the output, the better 2 images match.
'''
def SSD_sum(initial_image, x, y, target_image):
    result = 0;
    null = 0;
    '''We have to process the raw images first in order to calculate dot 
    product of two images. Basically we need to adjust their size'''
    ''' Prepares for the new processed images. '''
    init_top = x;
    init_down = null;
    init_left = y;
    init_right = null;
    
    target_top = 0;
    target_down = null;
    target_left = 0;
    target_right = null;
    
    ''' Setting the bottom edge for the new processed images'''
    if (initial_image.shape[0] <= target_image.shape[0] + x):
        init_down = initial_image.shape[0];
        target_down = initial_image.shape[0] - x;
    else:
        init_down = target_image.shape[0] + x;
        target_down = target_image.shape[0];
        
    ''' Setting the right edge for the new processed images'''
    if (initial_image.shape[1] <= target_image.shape[1] + y):
        init_right = initial_image.shape[1];
        target_right = initial_image.shape[1] - y;
    else:
        init_right = target_image.shape[1] + y;
        target_right = target_image.shape[1];
        
    '''Check if the type is correct '''
    if (type(init_top) == int and type(init_down) == int and type(init_left) == int and type(init_right) == int):
        processed_initial_image = initial_image[init_top:init_down, init_left:init_right];
        processed_target_image = target_image[target_top:target_down, target_left:target_right];
    else:
        return 0;
    
    processed_initial_image = processed_initial_image.astype(float);
    processed_target_image = processed_target_image.astype(float);

    squared_diff_matrix = 255.0 - np.sqrt((processed_target_image - processed_initial_image) ** 2);
    
    result = sum(squared_diff_matrix);
    
    return result;

'''
Input: initial_image(reference image), x, y(coordinates), target_image
Output: NCC of two images.
 ==> The larger the output, the better 2 images match.
'''
def NCC(initial_image, x, y, target_image):
    result = 0;
    null = 0;
    
    '''We have to process the raw images first in order to calculate dot 
    product of two images. Basically we need to adjust their size'''
    ''' Prepares for the new processed images. '''
    init_top = x;
    init_down = null;
    init_left = y;
    init_right = null;
    
    target_top = 0;
    target_down = null;
    target_left = 0;
    target_right = null;
    
    ''' Setting the bottom edge for the new processed images'''
    if (initial_image.shape[0] <= target_image.shape[0] + x):
        init_down = initial_image.shape[0];
        target_down = initial_image.shape[0] - x;
    else:
        init_down = target_image.shape[0] + x;
        target_down = target_image.shape[0];
        
    ''' Setting the right edge for the new processed images'''
    if (initial_image.shape[1] <= target_image.shape[1] + y):
        init_right = initial_image.shape[1];
        target_right = initial_image.shape[1] - y;
    else:
        init_right = target_image.shape[1] + y;
        target_right = target_image.shape[1];
    
    '''Check if the type is correct '''        
    if (type(init_top) == int and type(init_down) == int and type(init_left) == int and type(init_right) == int):
        processed_initial_image = initial_image[init_top:init_down, init_left:init_right];
        processed_target_image = target_image[target_top:target_down, target_left:target_right];
    else:
        return 0
    
    init_mean = np.mean(processed_initial_image);
    target_mean = np.mean(processed_target_image);
    
    top = ((processed_initial_image - init_mean) * (processed_target_image - target_mean)).sum();
    bottom_1 = (processed_initial_image - init_mean)**2;
    bottom_2 = (processed_target_image - target_mean)**2;
    bottom = np.sqrt(bottom_1.sum() * bottom_2.sum());
    result = top / bottom;

    return result;    
    
'''
Input: image: the image we try to combine the colors.
       is_NCC: if using SSD, the value is 0; is using NCC, then 1.
       is_large_image: if image is a large image, the value should be 1;else, 0.
Output: The color image we want to get.
'''
def A1(image, is_NCC = 0, is_large_image = 0):
    
    #These are margins for the no borders single-color pictures
    up_margin = 15;
    down_margin = 15;
    left_margin = 15;
    right_margin = 15;
    
    init_image = imread(image);
    
    '''Section for large images ---------------------------------------start'''
    if (is_large_image == 1):
        init_large_image = init_image / float(max(init_image.flatten()));
        init_large_image = init_large_image.astype(float) * 255.0;
        resize_rate = 400.0 / init_image.shape[1];
        init_image = imresize(init_image, resize_rate);     
    '''Section for large images ---------------------------------------end'''
    
    '''
    Separates the 3 different channel.
    '''    
    blue_image = init_image[0:init_image.shape[0]//3, :];
    green_image = init_image[init_image.shape[0]//3:init_image.shape[0] * 2//3, :];
    red_image = init_image[init_image.shape[0]*2//3:init_image.shape[0], :];
    
    '''Section for large images ---------------------------------------start'''
    if (is_large_image == 1):
        large_blue_image = init_large_image[0:init_large_image.shape[0]//3, :];
        large_green_image = init_large_image[init_large_image.shape[0]//3:init_large_image.shape[0] * 2//3, :];
        large_red_image = init_large_image[init_large_image.shape[0]*2//3:init_large_image.shape[0], :];
    '''Section for large images ---------------------------------------end'''
    
    '''
    Match the green channel to the blue one.
    Calculate the SSD_sum or NCC for -10 to 10 in x and y axis, find the max point
    in the figure.
    '''
    ''' This part in match green channel to the blue one '''
    min_SSD_1 = 0;
    x_min_SSD_1 = 0;
    y_min_SSD_1 = 0;
    special_green_image = green_image[0:green_image.shape[0]-30, 0:green_image.shape[1]-30]
    
    for x_1 in range(0, 10):
        for y_1 in range(0, 10):
            if (is_NCC == 0):
                SSD_cur_1 = SSD_sum(blue_image, x_1, y_1, special_green_image);
            elif (is_NCC == 1):
                SSD_cur_1 = NCC(blue_image, x_1, y_1, green_image);
            if (SSD_cur_1 > min_SSD_1):
                min_SSD_1 = SSD_cur_1;
                x_min_SSD_1 = x_1;
                y_min_SSD_1 = y_1;
                
    min_SSD_2 = 0;
    x_min_SSD_2 = 0;
    y_min_SSD_2 = 0;
    special_blue_image = blue_image[0:blue_image.shape[0]-30, 0:blue_image.shape[1]-30]
    
    for x_2 in range(0, 10):
        for y_2 in range(0, 10):
            if (is_NCC == 0):
                SSD_cur_2 = SSD_sum(green_image, x_2, y_2, special_blue_image);
            elif (is_NCC == 1):
                SSD_cur_2 = NCC(green_image, x_2, y_2, special_blue_image);
            if (SSD_cur_2 > min_SSD_2):
                min_SSD_2 = SSD_cur_2;
                x_min_SSD_2 = x_2;
                y_min_SSD_2 = y_2;
    
    '''Goal: Get 2 intermediate color images. '''
    '''Do the slicing to 2 color images. Two cases here: '''
    new_blue_image = 0;
    new_green_image = 0;
    
    if (min_SSD_1 >= min_SSD_2): # Case1, if green on top has larger 1-SSD or NCC
        new_blue_image = blue_image[x_min_SSD_1:blue_image.shape[0], y_min_SSD_1:blue_image.shape[1]];
        new_green_image = green_image[0:blue_image.shape[0]-x_min_SSD_1, 0:blue_image.shape[1]-y_min_SSD_1];   
        '''Section for large images ---------------------------------------'''
        if (is_large_image == 1):
            new_large_blue_image = large_blue_image[x_min_SSD_1//resize_rate:large_blue_image.shape[0], y_min_SSD_1//resize_rate:large_blue_image.shape[1]];
            new_large_green_image = large_green_image[0:large_blue_image.shape[0]-(x_min_SSD_1//resize_rate), 0:large_blue_image.shape[1]-(y_min_SSD_1//resize_rate)];   
        '''Section for large images ------------------------------------end'''
    else: # Case2, if blue on top is better
        new_green_image = green_image[x_min_SSD_2:green_image.shape[0], y_min_SSD_2:green_image.shape[1]];
        new_blue_image = blue_image[0:green_image.shape[0]-x_min_SSD_2, 0:green_image.shape[1]-y_min_SSD_2];
        
        '''Section for large images ---------------------------------------'''
        if (is_large_image == 1):
            new_large_green_image = large_green_image[x_min_SSD_2//resize_rate:large_green_image.shape[0], y_min_SSD_2//resize_rate:large_green_image.shape[1]];
            new_large_blue_image = large_blue_image[0:large_green_image.shape[0]-x_min_SSD_2//resize_rate, 0:large_green_image.shape[1]-y_min_SSD_2//resize_rate];      
        '''Section for large images ------------------------------------end'''
        
    ''' This part is match red channel to the new blue one(and green one) '''
    ''' Nedds two mode to cover -10 to 10 '''
    ''' 1st mode, red image on the top '''
    min_SSD_3 = 0;
    x_min_SSD_3 = 0;
    y_min_SSD_3 = 0;
    special_red_image = red_image[0:red_image.shape[0] - 30, 0:red_image.shape[1]-30]
    
    for x_3 in range(0, 10):
        for y_3 in range(0, 10):
            if (is_NCC == 0):
                SSD_cur_3 = SSD_sum(new_blue_image, x_3, y_3, special_red_image);
            elif (is_NCC == 1):
                SSD_cur_3 = NCC(new_blue_image, x_3, y_3, special_red_image);
            if (SSD_cur_3 > min_SSD_3):
                min_SSD_3 = SSD_cur_3;
                x_min_SSD_3 = x_3;
                y_min_SSD_3 = y_3;
                
    min_SSD_4 = 0;
    x_min_SSD_4 = 0;
    y_min_SSD_4 = 0;
    special_new_blue_image = new_blue_image[0:new_blue_image.shape[0] - 30, 0:new_blue_image.shape[1]-30]
    
    for x_4 in range(0, 10):
        for y_4 in range(0, 10):
            if (is_NCC == 0):
                SSD_cur_4 = SSD_sum(red_image, x_4, y_4, special_new_blue_image);
            elif (is_NCC == 1):
                SSD_cur_4 = NCC(red_image, x_4, y_4, special_new_blue_image);
            if (SSD_cur_4 > min_SSD_4):
                min_SSD_4 = SSD_cur_4;
                x_min_SSD_4 = x_4;
                y_min_SSD_4 = y_4;
                
    '''Goal: Get 3 final color images. '''
    '''Do the slicing to 3 color images. Two cases here: '''
    brand_new_blue_image = 0;
    brand_new_green_image = 0;
    brand_new_red_image = 0;
    if (min_SSD_3 >= min_SSD_4): #1st case: If red on top has a larger 1-SSD or NCC
        brand_new_blue_image = new_blue_image[x_min_SSD_3:new_blue_image.shape[0], y_min_SSD_3:new_blue_image.shape[1]];
        brand_new_green_image = new_green_image[x_min_SSD_3:new_blue_image.shape[0], y_min_SSD_3:new_blue_image.shape[1]];
        brand_new_red_image = red_image[0:new_blue_image.shape[0]-x_min_SSD_3, 0:new_blue_image.shape[1]-y_min_SSD_3];   
            
        '''Section forr large image ---------------------------------------'''
        if (is_large_image == 1):
            brand_new_large_blue_image = new_large_blue_image[x_min_SSD_3//resize_rate:new_large_blue_image.shape[0], y_min_SSD_3//resize_rate:new_large_blue_image.shape[1]];
            brand_new_large_green_image = new_large_green_image[x_min_SSD_3//resize_rate:new_large_blue_image.shape[0], y_min_SSD_3//resize_rate:new_large_blue_image.shape[1]];
            brand_new_large_red_image = large_red_image[0:new_large_blue_image.shape[0]-x_min_SSD_3//resize_rate, 0:new_large_blue_image.shape[1]-y_min_SSD_3//resize_rate];   
        '''Section forr large image ---------------------------------------'''
    else: # 2nd case: If blue on top has a larger 1-SSD or NCC
    
        '''Prepares the coordinates for the slicing '''
        Null = 0;
        
        blue_top = 0;
        blue_down = Null;
        blue_left = 0;
        blue_right = Null;
        
        red_top = x_min_SSD_4;
        red_down = Null;
        red_left = y_min_SSD_4;
        red_right = Null;
        
        ''' Setting the bottom edge for the new processed images'''
        if (red_image.shape[0] <= new_blue_image.shape[0] + x_min_SSD_4):
            red_down = red_image.shape[0];
            blue_down = red_image.shape[0] - x_min_SSD_4;
        else:
            red_down = new_blue_image.shape[0] + x_min_SSD_4;
            blue_down = new_blue_image.shape[0];
            
        ''' Setting the right edge for the new processed images'''
        if (red_image.shape[1] <= new_blue_image.shape[1] + y_min_SSD_4):
            red_right = red_image.shape[1];
            blue_right = red_image.shape[1] - y_min_SSD_4;
        else:
            red_right = new_blue_image.shape[1] + y_min_SSD_4;
            blue_right = new_blue_image.shape[1];
        
        '''Do the slicing here '''
        brand_new_red_image = red_image[red_top:red_down, red_left:red_right];
        brand_new_blue_image = new_blue_image[blue_top:blue_down, blue_left:blue_right];
        brand_new_green_image = new_green_image[blue_top:blue_down, blue_left:blue_right];
        
        '''Section for large image --------------------------------------'''
        if (is_large_image == 1):
            brand_new_large_red_image = large_red_image[red_top/resize_rate:red_down/resize_rate, red_left/resize_rate:red_right/resize_rate];
            brand_new_large_blue_image = new_large_blue_image[blue_top//resize_rate:blue_down//resize_rate, blue_left//resize_rate:blue_right//resize_rate];
            brand_new_large_green_image = new_large_green_image[blue_top//resize_rate:blue_down//resize_rate, blue_left//resize_rate:blue_right//resize_rate];
        '''Section for large image --------------------------------------'''
    
    ''' We will assign the colors here '''
    if (is_large_image == 0): #if this is a small image
        rgb_image = zeros(brand_new_blue_image.shape + (3,));
        rgb_image[:,:,0] = brand_new_red_image;
        rgb_image[:,:,1] = brand_new_green_image;
        rgb_image[:,:,2] = brand_new_blue_image;
    else: # IF this is a large image
        rgb_image = zeros(brand_new_large_blue_image.shape + (3,));
        rgb_image[:,:,0] = brand_new_large_red_image;
        rgb_image[:,:,1] = brand_new_large_green_image;
        rgb_image[:,:,2] = brand_new_large_blue_image;
    
    return rgb_image/255.0;
# 
figure(1); imshow(A1('00106v.jpg', 0))
#figure(2); imshow(A1('00888v.jpg', 0))
#figure(3); imshow(A1('00757v.jpg', 0))    
#    
figure(11); imshow(A1('00106v.jpg', 1))
#figure(12); imshow(A1('00888v.jpg', 1))
#figure(13); imshow(A1('00757v.jpg', 1))
#figure(4); imshow(A1('./00889v.jpg', 0))
#figure(14); imshow(A1('./00889v.jpg', 1))

figure(5); imshow(A1('./01880v.jpg', 0))
figure(15); imshow(A1('./01880v.jpg', 1))
#
#figure(6); imshow(A1('./01657v.jpg', 0))
#figure(16); imshow(A1('./01657v.jpg', 1))

#figure(8); imshow(A1('/home/will/Pyzo(csc320)/00911v.jpg', 0))
#figure(18); imshow(A1('/home/will/Pyzo(csc320)/00911v.jpg', 1))
figure(9); imshow(A1('./01031v.jpg', 0))
figure(19); imshow(A1('./01031v.jpg', 1))
#
png1 = A1('./00128u.png', 0, 1);
figure(7); imshow(png1)
png2 = A1('./00128u.png', 1, 1)
figure(17); imshow(png2);

png3 = A1('./00458u.png', 0, 1)
png4 = A1('./00458u.png',1,1)
figure(100); imshow(png3)
figure(101); imshow(png4)