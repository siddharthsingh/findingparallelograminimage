
import math
import numpy as np
from copy import copy
from matplotlib import pyplot as plt

#fnc to greate a guassian mask of given size
def create_guassian_mask(shape , mean , variance):
    x, y = np.meshgrid(np.linspace(-1,1,shape[0]), np.linspace(-1,1,shape[1]))
    d = np.sqrt(x**2 + y**2)
    g = np.exp(-((d-mean)**2 / ( 2.0*variance**2 )))
    return g
    

def is_image_grayscale(image):
    #check dimensionality of the image array, if it is a 3d array it means it is not grayscale
    if(len(image.shape)!=2 ):
        print("image or map not of proper dimension. Image and map should be 2D array")
        return False
    return True


    
def threshold_image(img , threshold , is_background_black):
    
    image = copy(img)
    if(not is_image_grayscale(image)):
        #image not grayscale
        return
    if(is_background_black):
        background = 0
        foreground = 1
    else:
        background = 1
        foreground = 0 
     #positions of all the points less than threshold   
    idx = image[:,:] > threshold
    image[idx] = foreground
    image[np.invert(idx)] = background

    return image




#detecting the threshold value
def find_threshold_using_peakiness(image):
    
    #####################################################################
    #################### Auto Thresholding ##############################
    #################### Peakiness Method ###############################
    #####################################################################
    ## In this method we compare every bin of histogram to every bin ahead of it.
    #create a historam list
    hist =  np.zeros(np.max(image)+1)
    image_rows_len = len(image)
    image_columns_len = len(image[0])
    for i in range(0 ,image_rows_len):
        for j in range(0, image_columns_len):
            #setting values to histogram list
	
            hist[image[i][j]] +=1


    #initialize the variables with -1 as any positive number can be actual value but negative value can't be
    max_peakiness = -1
    max_peakiness_maxima_one_pos = -1
    max_peakiness_maxima_two_pos = -1
    max_peakiness_minima_pos = -1
    # range of x is len(hist) - 2 because y goes to x+2
    for x in range(0,len(hist)-2):
        for y in range(x+2,len(hist)):
            if(x==y):
                continue
            if(abs(x-y)==1):
                continue
            array = hist[x:y+1]
            for i in range(x+1,y):

                if(hist[i]==0 or hist[x] == 0 or hist[y] == 0):
                    continue
                peakiness = min(hist[x] , hist[y])/hist[i]

                if(max_peakiness<peakiness):
                    max_peakiness = peakiness
                    max_peakiness_minima_pos = i
                    max_peakiness_maxima_one_pos = x
                    max_peakiness_maxima_two_pos = y

    return max_peakiness_minima_pos

#function to apply the given map of any size to the given image
def apply_map(image , image_map, is_binary ):
    mapped_image = np.zeros(image.shape)

    if(not is_image_grayscale(image)):
        #image not grayscale
        return
    
    n , m = image.shape
    map_rows , map_columns = image_map.shape
    if(map_rows%2==0 or map_columns%2 == 0):
        print("even map not supported")
        return
    image_rows_range_lower_limit = math.floor(map_rows/2)
    image_columns_range_lower_limit = math.floor(map_columns/2)
    image_rows_range = range(int(image_rows_range_lower_limit) , int(n-image_rows_range_lower_limit))
    image_columns_range = range(int(image_columns_range_lower_limit) , int(m-image_columns_range_lower_limit))
    if(is_binary):
        for i in image_rows_range:
            for j in image_columns_range:
                mapped_image[i][j] = sum(sum(np.multiply(image[i-int(np.floor(float(map_rows)/2)):int(i+np.ceil(float(map_rows)/2)),j-int(np.ceil(float(map_columns)/2)):j+int(np.ceil(float(map_columns)/2))] , image_map)))
    else:
        for i in image_rows_range:
            for j in image_columns_range:
                #print(map_rows , int(np.floor(map_rows/2)) , int(np.ceil(map_rows/2)))
                mapped_image[i][j] = round(sum(sum(np.multiply(image[i-int(np.floor(float(map_rows)/2)):i+int(np.ceil(float(map_rows)/2)),j-int(np.floor(float(map_columns)/2)):j+int(np.ceil(float(map_columns)/2))] , image_map))))
    
        
    return mapped_image


def histogram_equalization(image , new_histogram_range):
    output_image = np.zeros(image.shape)
    min_image = np.min(image)
    max_image = np.max(image)
    temp = (new_histogram_range[1] - new_histogram_range[0])/(max_image - min_image)
    for i in range(0 , len(image)):
        for j in range(0 , len(image[0])):
           output_image[i][j] = temp*(image[i][j] - min_image) + new_histogram_range[0]
    return output_image

#######edge operator operator
def edge_operator(image , type_of_operator ):
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    img2 = np.zeros(image.shape)
    gradient_angle = np.zeros(image.shape)
    n , m = image.shape

    
    if(type_of_operator=="sobel" or type_of_operator=="sobels" ):
     
        edge_operator_x = np.array([[-1,0,1], [-2,0,2] ,[-1,0,1]])
        edge_operator_y = np.array([[-1,-2,-1], [0,0,0] ,[1,2,1]])
    elif(type_of_operator=="log" or type_of_operator=="LOG" ):
        edge_operator_x = np.array([[0,0,-1,0,0], [0,-1,-2,-1,0] ,[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
        edge_operator_y = np.array([[-1,-2,-1], [0,0,0] ,[1,2,1]])
    elif(type_of_operator=="prewitt"):

        edge_operator_x = np.array([[-1,0,1], [-1,0,1] ,[-1,0,1]])
        edge_operator_y = np.array([[-1,-1,-1], [0,0,0] ,[1,1,1]])
    else:
        print("edge operator not recognized.")
        return None

    #gradient along x and y direction , actually it is in i, j co-ordinate system
    gx = apply_map(image , edge_operator_x , False)
    gy = apply_map(image , edge_operator_y , False)

    img2 = np.round(np.sqrt(gx**2 + gy**2))
    
    #this line may cause division by zero error as dx might be zero in (dy/dx)
    #print(np.arctan(np.divide(gy,gx)))
    
    #gradient_angle = np.arctan(np.divide(gy,gx))
    for i in range(len(gy)):
        for j in range(len(gy[0])):
            if gx[i][j] == 0:
                if gy[i][j] == 0:
                   gradient_angle[i][j] = 0
                elif gy[i][j] > 0: gradient_angle[i][j] = float(90)
                else:gradient_angle[i][j] = float(270)
            else:
                gradient_angle[i][j] = np.arctan(gy[i][j]/gx[i][j])

    for i in range(len(gradient_angle)):
        for j  in range(len(gradient_angle[0])):
            angle = math.degrees(gradient_angle[i][j])
            if angle  < 0: 
                angle = 360+angle# angle is -ve so adding it would subtract it from 360
            gradient_angle[i][j] = angle 

    return img2 , gradient_angle

#quantizing angles in diff sectors
def quantize_gradient_angle_to_secotrs(gradient_angle):
    quantized_gradient_angle = np.zeros(gradient_angle.shape)
    for i in range(0 , len(gradient_angle)):
        for j in range(0 , len(gradient_angle[0])): 
            
            if( math.isnan(gradient_angle[i][j])):
                quantized_gradient_angle[i][j] = 0
            elif(gradient_angle[i][j]>337.5):
                quantized_gradient_angle[i][j] = 0
            elif(gradient_angle[i][j]>=0 and gradient_angle[i][j]<=22.5):
                quantized_gradient_angle[i][j] = 0
            elif(gradient_angle[i][j]>22.5 and gradient_angle[i][j]<=67.2):
                quantized_gradient_angle[i][j] = 1
            elif(gradient_angle[i][j]>67.5 and gradient_angle[i][j]<=112.5):
                quantized_gradient_angle[i][j] = 2
            elif(gradient_angle[i][j]>112.5 and gradient_angle[i][j]<=157.5):
                quantized_gradient_angle[i][j] = 3
            elif(gradient_angle[i][j]>157.5 and gradient_angle[i][j]<=202.5):
                quantized_gradient_angle[i][j] = 0
            elif(gradient_angle[i][j]>202.5 and gradient_angle[i][j]<=247.5):
                quantized_gradient_angle[i][j] = 1
            elif(gradient_angle[i][j]>247.5 and gradient_angle[i][j]<=292.5):
                quantized_gradient_angle[i][j] = 2
            elif(gradient_angle[i][j]>292.5 and gradient_angle[i][j]<=337.5):
                quantized_gradient_angle[i][j] = 3
            
    return quantized_gradient_angle


def non_maxima_suppression(image_gardient , quantized_gradient_angle):
    
    image_gardient_rows , image_gardient_columns = image_gardient.shape
    output_image = np.zeros(image_gardient.shape)
    for i in range(1,image_gardient_rows-1):
        for j in range(1,image_gardient_columns-1):
            if(image_gardient[i][j]) == 0 : continue
            
            #if(math.isnan(quantized_gradient_angle[i][j])): continue
            
            if(quantized_gradient_angle[i][j] == 0):
                if(image_gardient[i][j] < image_gardient[i][j-1] or 
                   image_gardient[i][j] < image_gardient[i][j+1]):
                    continue
                else:output_image[i][j] = image_gardient[i][j]
                
            
            if(quantized_gradient_angle[i][j] == 1):
                if(image_gardient[i][j] < image_gardient[i-1][j+1] or 
                   image_gardient[i][j] < image_gardient[i+1][j-1]):
                    continue
                else:output_image[i][j] = image_gardient[i][j]
                
            
            if(quantized_gradient_angle[i][j] == 2):
                if(image_gardient[i][j] < image_gardient[i-1][j] or 
                   image_gardient[i][j] < image_gardient[i+1][j]):
                    continue
                else:output_image[i][j] = image_gardient[i][j]
                
            
            if(quantized_gradient_angle[i][j] == 3):
                if(image_gardient[i][j] < image_gardient[i-1][j-1] or 
                   image_gardient[i][j] < image_gardient[i+1][j+1]):
                    continue
                else:output_image[i][j] = image_gardient[i][j]
                
     
    return output_image

def gmaximum(det, phase):
  gmax = np.zeros(det.shape)
  for i in xrange(gmax.shape[0]):
    for j in xrange(gmax.shape[1]):
      if phase[i][j] < 0:
        phase[i][j] += 360

      if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
        # 0 degrees
        if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
          if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
            gmax[i][j] = det[i][j]
        # 45 degrees
        if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
          if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
            gmax[i][j] = det[i][j]
        # 90 degrees
        if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
          if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
            gmax[i][j] = det[i][j]
        # 135 degrees
        if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
          if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
            gmax[i][j] = det[i][j]
  return gmax


def canny_edge_detection(image):
    
    g = create_guassian_mask([3,3] , 0.0 , 1)  
    img_after_gaussian_mask = (apply_map(image , g , False) / sum(sum(g)))
    print(img_after_gaussian_mask.shape, image.shape)
    #sobels operator returns the gradient magnitude array and the gradient direction array
    #the apply map function could also be used
    #calculating the gradients and the angles
    gx = np.zeros(img_after_gaussian_mask.shape)
    gy = np.zeros(img_after_gaussian_mask.shape)
    gxy = np.zeros(img_after_gaussian_mask.shape)
    theta = np.zeros(img_after_gaussian_mask.shape)
    rows , columns = img_after_gaussian_mask.shape
    print(rows)
    edge_operator_x = np.array([[-1,0,1], [-2,0,2] ,[-1,0,1]])
    edge_operator_y = np.array([[-1,-2,-1], [0,0,0] ,[1,2,1]])
    for i in range(1 , rows-2):
        for j in range(1 , columns-1):
            #print(i)
            gx[i][j] = (-1*(img_after_gaussian_mask[i-1][j-1]))+(-2*(img_after_gaussian_mask[i][j-1]))+(-1*(img_after_gaussian_mask[i+1][j-1]))+(1*(img_after_gaussian_mask[i-1][j+1]))+(2*(img_after_gaussian_mask[i][j+1]))+(1*(img_after_gaussian_mask[i+1][j+1]))
            gy[i][j] = (-1*(img_after_gaussian_mask[i-1][j-1]))+(-2*(img_after_gaussian_mask[i-1][j]))+(-1*(img_after_gaussian_mask[i-1][j+1]))+(1*(img_after_gaussian_mask[i+1][j-1]))+(2*(img_after_gaussian_mask[i+1][j]))+(1*(img_after_gaussian_mask[i+1][j+1]))
            gxy[i][j] = np.sqrt(gx[i][j]**2 + gy[i][j]**2)
          
    for i in range(len(gy)):
        for j in range(len(gy[0])):
            if gx[i][j] == 0:
                if gy[i][j] == 0:
                   theta[i][j] = 0
                elif gy[i][j] > 0: theta[i][j] = float(90)
                else:theta[i][j] = float(270)
            else:
                theta[i][j] = np.arctan(gy[i][j]/gx[i][j])
    #if an angle is negative make it positive
    for i in range(len(theta)):
        for j  in range(len(theta[0])):
            angle = math.degrees(theta[i][j])
            if angle  < 0: 
                angle = 360+angle# angle is -ve so adding it would subtract it from 360
            theta[i][j] = angle 




    #image_after_sobels_operator , theta = edge_operator(img_after_gaussian_mask , "sobels")
    quantized_theta = quantize_gradient_angle_to_secotrs(theta)
    #print(theta , quantized_theta)
    #non maxima suppression function can also be used here. 
    image_after_non_maxima_suppression = gmaximum(gxy ,theta)

    return image_after_non_maxima_suppression        


