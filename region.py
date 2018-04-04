
from scipy import ndimage
import cv2
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
from collections import Counter
import heapq
import itertools
from sklearn.cluster import KMeans
from copy import copy

#####################################################################
#################### REGION SEGMENTATION ############################
#####################################################################


def segment_region(img):
    labels = img*0
    n , m = img.shape
    #To-Do:create a list of tuples of int,int to store same labels
    eqivalenceList = []
    #increment label number only if you assign a new label
    label = 1
    for i in range(0 ,n):
        for j in range(0, m):
            if(img[i][j]==1):
                #check if row above exists
                if(i>0 and j>0):
                    #we are not in the top row or leftmost column, label pixel if 1
                    if(img[i][j]==1):
                        #check if the pixel above it has a label
                        if(labels[i-1][j]!=0):
                            labels[i][j] = labels[i-1][j]
                        else:
                            #pixel above doesn't have a label, check left neibhor pixel
                            if(labels[i][j-1]!=0):
                                labels[i][j]=labels[i][j-1]
                            else:
                                #it doesn't have a top or left neighboring labelled pixel so assign a new pixel
                                labels[i][j]=label
                                label +=1
                        #if both top , left nieghboring pixels are there, we have to insert 
                        #them in equivalence table
                        if(labels[i-1][j]!=0 and labels[i][j-1]!=0 and labels[i][j-1] != labels[i-1][j]):
                            #check if this pair of values are not already in the list
                            inserted = False
                            for evalList in eqivalenceList:
                                if( labels[i-1][j] in evalList ):
                                    evalList.append(labels[i][j-1])
                                    inserted = True
                                    break
                                elif(labels[i][j-1] in evalList):
                                    evalList.append(labels[i-1][j])
                                    inserted = True
                                    break
                            if(not inserted):
                                eqivalenceList.append([labels[i][j-1] ,labels[i-1][j] ])



                else:
                    #we are in the top row or the top column
                    #check if we are in the top row. If we are in the top row and there is a pixel on the left
                    #that is labelled , then we can just copy the label.
                    if(i==0 and j>0):
                        if(labels[i][j-1]!=0):
                            labels[i][j] = labels[i][j-1]
                    if(i>0 and j==0):
                        if(labels[i-1][j]!=0):
                            labels[i][j] = labels[i-1][j]
                    #case 0,0
                    if(i==0 and j==0):
                        if(img[i][j]==1):
                            labels[i][j]=label
                            label = label+1

    #length of equivalence list does not tell the number of regions. background that are one pixel in area
    #or other regions in which only one label is set to all the pixels in the regions will not be in the 
    #equivalance list
    #print(len(eqivalenceList))

    for evalList in eqivalenceList:
        #print(min(evalList))
        for i in range(0 ,n):
            for j in range(0, m):
                if(labels[i][j] in evalList):
                    labels[i][j] = min(evalList)

    # to get the total number of regions see the number of different values in labels array
    #background will be given label 0
    diffLabels = []
    for i in range(0 ,n):
        for j in range(0, m):
            if(labels[i][j] not in diffLabels):
                diffLabels.append(labels[i][j])
    #print(diffLabels)
    
#     plt.imshow(img ,  cmap='gray')
#     plt.show()
    return diffLabels , labels






#####################################################################
##################### Findind Areas ##################################
#####################################################################
def calculate_area(image , diff_labels  , labels):
    Areas = [0]*len(diff_labels)
    n , m = image.shape
    for x in range(0 ,len(diff_labels)):
        for i in range(0 ,n):
            for j in range(0, m):
                if(labels[i][j] == diff_labels[x]):
                    Areas[x] += 1
    return Areas

#####################################################################
######Size Filter-Deleting Areas with minimum Area = x pixels#######
#####################################################################
def area_filter(img , min_area):
    if(min_area == 0): return img
    image = copy(img)
    n,m = image.shape
    diff_labels , labels = segment_region(image)
    Areas = calculate_area(image , diff_labels , labels)
    for x in range(1 ,len(diff_labels)):
        for i in range(0 ,n):
            for j in range(0, m):
                if(labels[i][j] == diff_labels[x] and Areas[x]<min_area):
                    image[i][j] = 0
    #plt.imshow(image ,  cmap='gray')
    #plt.show()
    return image

    

         

