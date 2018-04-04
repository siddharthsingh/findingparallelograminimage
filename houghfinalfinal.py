
# coding: utf-8

# In[163]:

#parameters for this code
step_size_p  = 2
step_size_theta = 3
#to make code run faster make this True
resize_image = True
resize_factor = 0.4
grayscale_to_binary_threshold = 40
accumulator_threhold = 0.4 # of the maximum
image_name = "test3" # put both raw and jpg image in the folder to run this code



import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import edgedetection as ed
np.set_printoptions(threshold=np.nan)
from scipy.misc import imresize

from copy import copy




#reading the image
image_file_name = image_name+".raw"
img_file = open(image_file_name , 'rb')
img_color = np.fromfile(img_file, dtype='uint8')
rows,columns, c = cv2.imread(image_name+".jpg").shape #this is just to get the size of the image
img_color= img_color.reshape([rows,columns*3])


img_gray_man = np.zeros([rows,columns])


color_i_temp = np.zeros( (rows,columns, c), np.uint8)
#converting the image to grayscale
for i in range(0,rows):
    for j in range(0,columns):
        img_gray_man[i][j] = (0.3*img_color[i][j*3] +0.59*img_color[i][j*3+1] +0.11*img_color[i][j*3+2])

#resize the image
if(resize_image):
    img_gray_man1 = np.array(img_gray_man, dtype=np.uint8)
    img_gray_man = imresize(img_gray_man1, resize_factor, 'bilinear')
    rows,columns = img_gray_man.shape
#perform canny edge detection
img_after_canny = ed.canny_edge_detection(img_gray_man)

# threshold grascale image to get the binary image
binary_image = ed.threshold_image(img_after_canny,grayscale_to_binary_threshold, True)

#removing the boundary lines
for j in range(0,columns):
    binary_image[1][j]=0
    binary_image[rows-2][j]=0
for i in range(0,rows):
    binary_image[i][1]=0
    binary_image[i][columns-2]=0

plt.imshow(binary_image , cmap='gray')
plt.show()
#create a list of thetas
def get_theta_list(step_size):
    if 360 % step_size != 0:
        print("step_size not a factor of 180")
    no_of_steps = int(round(360/step_size))

    return np.linspace(0 , 360 ,no_of_steps )+(round(step_size/2))



def calculatep(theta , x , y ):
    return x*math.cos(math.radians(theta)) + y*math.sin(math.radians(theta))

thetas = get_theta_list(step_size_theta)

 

#used to create an empty accumulator array 
def create_accumulator(len_thetas , step_size_p ):
    diag_len = np.ceil(np.sqrt(columns * columns + rows * rows))  
    
    p_list_len = round(diag_len/step_size_p)
    return np.zeros((2*int(p_list_len)+1,len_thetas+1))
print(thetas , len(thetas))


diag_len = np.ceil(np.sqrt(columns * columns + rows * rows))  
def increment_accumulator(image , thetas , theta_step_size  = step_size_theta, p_step_size  = step_size_p):
    accumulator_positions = {}
    n,m = image.shape
    
    
    accumulator  = create_accumulator(len(thetas) , p_step_size)

    accumulator_rows , accumulator_columns = accumulator.shape
    print(accumulator.shape)
    # this line get the index of all the points that are non zero in the binary image
    for i in range(rows):
        print i , rows
        for j in range(columns):
            if binary_image[i][j]==1:
                for theta in thetas:
                    theta = int(round(theta))
                    theta_position_in_accumulator = int(round((theta)/step_size_theta))-1

                    result = int(round((j*np.cos(math.radians(theta)) + i*np.sin(math.radians(theta)))+diag_len)/step_size_p)-1
                    accumulator[result][theta_position_in_accumulator] += 1
                    #check if the point is already at this loc in dictionary
                    if((result,theta_position_in_accumulator) in accumulator_positions.keys()):
                        accumulator_positions[result,theta_position_in_accumulator].append((i,j))
                    else:
                        accumulator_positions[result,theta_position_in_accumulator] = [(i,j)]

    return accumulator , accumulator_positions

#this function is used to threshold the accumulator
def threshold_accumulator(accumulator , accumulator_positions , threshold ):
    accumulator_rows , accumulator_columns = accumulator.shape
    a = [accumulator_positions[i,j] for i in range(0,accumulator_rows) for j in range(0, accumulator_columns) 
           if accumulator[i][j] > threshold]
    return a

#used to cread the image from the accumulator    
def create_image_from_accumulator( accumulator , accumulator_positions , threshold , image_shape):
    list_of_points = threshold_accumulator(accumulator , accumulator_positions , threshold)
    output_img = np.zeros(image_shape)
    if(not list_of_points): 
        print("null list")
    for list1 in list_of_points:
        for element_tupple in list1:
            output_img[element_tupple[0] , element_tupple[1]] = 1
    return output_img   

#calling this func sets value to the accumulator array and also returns the points corresponding to 
#each cell in accumulator
man_image_accumulator , man_image_accumulator_pos = increment_accumulator(binary_image , thetas)



print(diag_len)


#max value of the accumulator array
max_acc_man = np.max(man_image_accumulator)
print(max_acc_man)


# In[165]:



# In[ ]:


from scipy.spatial import distance
t1m = accumulator_threhold*max_acc_man

local_maxima = copy(man_image_accumulator)
                                        
#positions of the elements in accumulatar array which are greater than threshold
idx = man_image_accumulator[:,:] < t1m
local_maxima[idx] = 0


#creating a new dictionary of points which correspond to value greater than threshold in accumulator array
man_image_accumulator_pos2 = {}
list_of_points = []
for key , value in man_image_accumulator_pos.iteritems():
    
    if local_maxima[key[0]][key[1]] > t1m:
        if(len(value) >20):
            man_image_accumulator_pos2[key[0] ,key[1]] = value
            list_of_points.append(value)

#this is to check which all points we got after thresholding           
man_img_t1 = np.zeros(binary_image.shape)
if(not list_of_points): 
    print("null list")
for list1 in list_of_points:
    for element_tupple in list1:
        man_img_t1[element_tupple[0] , element_tupple[1]] = 1

plt.figure(figsize=(14, 12))
plt.imshow(man_img_t1 , cmap='gray')
plt.savefig( "Image1_after_hough.png" , bbox_inches='tight')
plt.show()
      
# print(len(man_image_accumulator_pos) , )

#list to store all the corner points
points_of_pg = []

#store the pair of lines so that we don't use it again
line12 = []
max_len_keys = len(man_image_accumulator_pos2)
#this is just to check the progress
iii = 0

#find two pair of parallel lines and then for every pair of parallel lines ind two more parallel lines
for key1 in man_image_accumulator_pos2.keys():
    print("progress :"+ str(iii)+"/"+str(max_len_keys))
    iii +=1
    for key2 in man_image_accumulator_pos2.keys():
        if key1 == key2: continue
        theta_val1 = ((key1[1]+1)*step_size_theta)
        theta_val2 = ((key2[1]+1)*step_size_theta)
        theta_val11 = int(theta_val1)
        theta_val22 = int(theta_val2)
        

        p1 = (key1[0]+1)*step_size_p - diag_len
        p2 = (key2[0]+1)*step_size_p - diag_len
        #check if the are certain distance apart
        if(np.abs(p1 - p2) < 30):continue
        if([min(key1,key2), max(key1,key2)] in line12):continue
        line12.append([min(key1,key2), max(key1,key2)])
        #the angle of two lines should not be very different
        if abs(theta_val1  - theta_val2 ) < 5:
#             print(theta_val1 , theta_val2)
            # parallel lines
            theta_val1  = np.deg2rad(theta_val1)
            theta_val2  = np.deg2rad(theta_val2)
            line34 = []
            for key3 in man_image_accumulator_pos2.keys():
                for key4 in man_image_accumulator_pos2.keys():
                    if key3 == key4 or key1 == key3 or key1==key4 or key2==key3 or key2 == key4: continue
                    theta_val3 = ((key3[1]+1)*step_size_theta)
                    theta_val4 = ((key4[1]+1)*step_size_theta)

                    if([min(key3,key4), max(key3 ,key4)] in line34):continue
                    line34.append([min(key3,key4), max(key3 , key4)])
                    #the angle of two lines should not be very different
                    if np.abs(theta_val3  - theta_val4 ) < 5:
                        
                        #check if the non parallel lines are atleast 50 degree apart
                        if( abs(theta_val1 - theta_val3) < 50 or
                        abs(theta_val1 - theta_val4) < 50 or
                        abs(theta_val2 - theta_val3) < 50 or
                        abs(theta_val2 - theta_val4) < 50):continue
                            

                        p3 = (key3[0]+1)*step_size_p - diag_len
                        p4 = (key4[0]+1)*step_size_p - diag_len
                        #print p3 , p4 , "works1"

                        if(np.abs(p3 - p4) < 30):continue
                        theta_val3  = np.deg2rad(theta_val3)
                        theta_val4  = np.deg2rad(theta_val4)
            
                        #intersection points of the four lines
                        try:
                            y13 = (p1/np.cos(theta_val1) - p3/np.cos(theta_val3))/(np.tan(theta_val1) - np.tan(theta_val3)) 
                            x13 = (p1 - y13*np.sin(theta_val1))/(np.cos(theta_val1))

                            y14 = (p1/np.cos(theta_val1) - p4/np.cos(theta_val4))/(np.tan(theta_val1) - np.tan(theta_val4)) 
                            x14 = (p1 - y14*np.sin(theta_val1))/(np.cos(theta_val1))


                            y23 = (p2/np.cos(theta_val2) - p3/np.cos(theta_val3))/(np.tan(theta_val2) - np.tan(theta_val3)) 
                            x23 = (p2 - y23*np.sin(theta_val2))/(np.cos(theta_val2))

                            y24 = (p2/np.cos(theta_val2) - p4/np.cos(theta_val4))/(np.tan(theta_val2) - np.tan(theta_val4)) 
                            x24 = (p2 - y24*np.sin(theta_val2))/(np.cos(theta_val2))

                        
                        
                        
                            y13 = int(np.round(y13))
                            x13 = int(np.round(x13))

                            y14 = int(np.round(y14))
                            x14 = int(np.round(x14))

                            y23 = int(np.round(y23))
                            x23 = int(np.round(x23))

                            y24 = int(np.round(y24))
                            x24 = int(np.round(x24))
                        except:
                            continue
#                         print "works2"
                        
                        #check if the points found are inside the image
                        if (y13 >= 0 and y13 <= rows and
                            y23 >= 0 and y23 <= rows and
                            y14 >= 0 and y14 <= rows and
                            y24 >= 0 and y24 <= rows):
                            
                            if (x13 >= 0 and x13 <= columns and
                            x23 >= 0 and x23 <= columns and
                            x14 >= 0 and x14 <= columns and
                            x24 >= 0 and x24 <= columns):
                                
                                if (
                                    np.abs(y13 - y23) < 10 and
                                    np.abs(x13 - x14 ) < 10): continue


                                temp1 = man_image_accumulator_pos2[key1]
                                temp2 = man_image_accumulator_pos2[key2]
                                temp3 = man_image_accumulator_pos2[key3]
                                temp4 = man_image_accumulator_pos2[key4]
                            
                    
                                count1 = 0
                                count2 = 0
                                count3 = 0
                                count4 = 0
                                
                                d1 = distance.euclidean((y13,x13),(y14 , x14))
                                d2 = distance.euclidean((y23,x23),(y24 , x24))
                                d3 = distance.euclidean((y13,x23),(y23 , x23))
                                d4 = distance.euclidean((y14,x14),(y24 , x24)) 
                                #checking if the points of intersection actually a line.
                                for pt in temp1:    
                                    if (pt[1] > min(x13,x14) and pt[0] > min(y13,y14)) and (pt[1] < max(x13,x14) and pt[0] < max(y13,y14)):
                                        count1 +=1 
                                if(d1/count1>4):continue
#                                 print "works4"

                                for pt in temp2:
                                    if (pt[1] > min(x23,x24) and pt[0] > min(y23,y24)) and (pt[1] < max(x23,x24) and pt[0] < max(y23,y24)):
                                        count2 +=1
                                if(d2/count2>4):continue
#                                 print "works5"

                                for pt in temp3:
                                    if (pt[1] > min(x13,x23) and pt[0] > min(y13,y23)) and (pt[1] < max(x13,x23) and pt[0] < max(y13,y23)):
                                        count3 +=1 
                                if(d3/count3>4):continue
#                                 print "works6"
                                #check to make sure the one of the sides is not very short
                                if(min(d1,d2)/max(d1,d2) < 0.6):continue
                                if(min(d3,d4)/max(d3,d4) < 0.6):continue
                        
#                                 print("7")
                    
#                                 for pt in temp4:
#                                     if (pt[1] > min(x14,x24) and pt[0] > min(y14,y24)) and (pt[1] < max(x14,x24) and pt[0] < max(y14,y24)):
#                                         count4 +=1
#                                 if(count4<10):continue
#                                 print "works7"

                                points_of_pg.append([(int(y13),int(x13)) , (int(y14),int(x14)) ,(int(y24),int(x24)),
                                                     (int(y23),int(x23))])

        


print("tgdh",len(points_of_pg))


print(points_of_pg)

ijj = 0
for point_of_pg in points_of_pg:

    color_i_temp =  cv2.line(color_i_temp ,(point_of_pg[0][1] , point_of_pg[0][0] ),(point_of_pg[1][1], point_of_pg[1][0] ),(255,0,0),1)
    color_i_temp =  cv2.line(color_i_temp ,(point_of_pg[1][1] , point_of_pg[1][0] ),(point_of_pg[2][1], point_of_pg[2][0] ),(255,0,0),1)
    color_i_temp =  cv2.line(color_i_temp ,(point_of_pg[2][1] , point_of_pg[2][0] ),(point_of_pg[3][1], point_of_pg[3][0] ),(255,0,0),1)
    color_i_temp =  cv2.line(color_i_temp ,(point_of_pg[3][1] , point_of_pg[3][0] ),(point_of_pg[0][1], point_of_pg[0][0] ),(255,0,0),1)


# In[110]:


# cv2.imshow('image',img_color)
plt.figure(figsize=(14, 12))
plt.imshow(color_i_temp  , cmap='gray')
plt.show()


# In[156]:


#removing almost same corners
points_of_pg2 = copy(points_of_pg)
# for p1 in points_of_pg:
#     for p2 in points_of_pg:
#         if p1 == p2: continue
#         d1 = distance.euclidean(p1[0],p2[0])
#         d2 = distance.euclidean(p1[1],p2[1])
#         d3 = distance.euclidean(p1[2],p2[2])
#         d4 = distance.euclidean(p1[3],p2[3])
        
#         if(d1+d2+d3+d4) < 40:
#             try:
#                 points_of_pg2.remove(p2)
#             except:pass
            

print(points_of_pg2)
#reading the image
image_file_name = image_name+".raw"
img_file = open(image_file_name , 'rb')
img_color = np.fromfile(img_file, dtype='uint8')
rows,columns, c = cv2.imread(image_name+".jpg").shape #this is just to get the size of the image
img_color= img_color.reshape([rows,columns*3])



color_i_temp = np.zeros( (rows,columns, c), np.uint8)

for i in range(0,rows):
    for j in range(0,columns):
        color_i_temp[i][j][0] = img_color[i][j*3]
        color_i_temp[i][j][1] = img_color[i][j*3+1]
        color_i_temp[i][j][2] = img_color[i][j*3+2]


#drawing on original image
for point_of_pg in points_of_pg2:

    if(not resize_image):
       
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[0][1])) , int(round(point_of_pg[0][0])) ),(int(round(point_of_pg[1][1])), int(round(point_of_pg[1][0])) ),(255,0,0),1)
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[1][1])) , int(round(point_of_pg[1][0])) ),(int(round(point_of_pg[2][1])), int(round(point_of_pg[2][0])) ),(255,0,0),1)
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[2][1])) , int(round(point_of_pg[2][0])) ),(int(round(point_of_pg[3][1])), int(round(point_of_pg[3][0])) ),(255,0,0),1)
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[3][1])), int(round(point_of_pg[3][0])) ),(int(round(point_of_pg[0][1])), int(round(point_of_pg[0][0])) ),(255,0,0),1)
    else:
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[0][1]/resize_factor)) , int(round(point_of_pg[0][0]/resize_factor)) ),(int(round(point_of_pg[1][1]/resize_factor)), int(round(point_of_pg[1][0]/resize_factor)) ),(255,0,0),1)
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[1][1]/resize_factor)) , int(round(point_of_pg[1][0]/resize_factor)) ),(int(round(point_of_pg[2][1]/resize_factor)), int(round(point_of_pg[2][0]/resize_factor)) ),(255,0,0),1)
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[2][1]/resize_factor)) , int(round(point_of_pg[2][0]/resize_factor)) ),(int(round(point_of_pg[3][1]/resize_factor)), int(round(point_of_pg[3][0]/resize_factor)) ),(255,0,0),1)
        color_i_temp =  cv2.line(color_i_temp ,(int(round(point_of_pg[3][1]/resize_factor)), int(round(point_of_pg[3][0]/resize_factor)) ),(int(round(point_of_pg[0][1]/resize_factor)), int(round(point_of_pg[0][0]/resize_factor)) ),(255,0,0),1)

plt.figure(figsize=(14, 12))
plt.imshow(color_i_temp  , cmap='gray')
plt.show()

for i in range(0,rows):
    for j in range(0,columns):
        color_i_temp[i][j][0] = img_color[i][j*3]
        color_i_temp[i][j][1] = img_color[i][j*3+1]
        color_i_temp[i][j][2] = img_color[i][j*3+2]

