#from sklearn.mixture import GMM

import sklearn.mixture
import numpy as np
import math
import cv2
from collections import defaultdict 
import time
import maxflow
from pylab import *
from numpy import *
from PIL import Image
import argparse
import progressbar
from time import sleep


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
img = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)

img_size =(len(img),len(img[0]))
data_file = open("train_data.txt","r") 
data=[]
output=[]
#Rp=np.zeros(img_size)

PF=np.zeros(img_size)
PB=np.zeros(img_size)
K=0.5
lamda=0.0001
#------------------------- Trainning to get reginal and boundary properties ------------------------------------#

print("\n-Training Mixture")
start=time.time()
while (data_file.readline()):
    
    line=data_file.readline()
    if (len(line)>0):
        nums = line.split(' ')
        intensity=int(nums[0])
        is_foreground=int(line.split(' ')[1][0])
        data.append([intensity])
        output.append([is_foreground])
gmm = sklearn.mixture.GaussianMixture(n_components=2).fit(data)
end=time.time()
print("-Done in "+str(end-start))
#--------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------#
print("\n-Calculating foreground/background probability for each pixel")
bar = progressbar.ProgressBar(maxval=len(img), \
widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#start calculating foreground/background probability for each 
start=time.time()
bar.start()
for i in range(len(img)):

    for j in range(len(img[0])):
        result = gmm.predict_proba([ [img[i,j]] ])[0]
        PF[i][j]=round(result[0],5)
        PB[i][j]=round(result[1],5)
    bar.update(i+1)
bar.finish()
#print Rp
end=time.time()
print("-Done in "+str(end-start))
#--------------------------------------------------------------------------------#



print("\n-Generating Bpq")
#Calculating Regional Property
Bpq = cv2.Canny(img,img_size[0],img_size[0])
Bpq =  np.absolute(    cv2.distanceTransform(Bpq,cv2.DIST_L2,5)  )
print("-Done")
print("\nInitializing graph")
print("-Done in "+str(end-start))


#---------------------------------Constucting graph and flow ------------------------------------#

def get_node_number(i,j):
	return img_size[0]*i+j
def add_edge(a,b,i,j,weight,g):

    source=get_node_number(i,j)
    dest=get_node_number(a,b)
    weight*=10
    g.add_edge(source,dest,weight, 0)

def add_t_links(i,j,g):

    node = get_node_number(i,j)
    g.add_tedge(node,PF[i][j]*5, PB[i][j]*5 )
 

def add_neighborhood_edges(i,j,g):
   
    #w=10
    k=10
    s=100
 #   print img[i][j]
    if (i-1>=0): 
        diff = img[i][j]-img[i-1][j]
        w = k*exp(-(abs(diff**2)/s))

        add_edge(i,j,i-1,j,w,g)
        if (j-1>=0): 
  
            diff = img[i][j]-img[i-1][j-1]
            w =  k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i-1,j-1,w,g)
        if (j+1<img_size[0]):
    
            diff = img[i][j]-img[i-1][j+1]
            w = k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i-1,j+1,w,g)
    if (i+1<img_size[0]):

        diff = img[i][j]-img[i+1][j]
        w = k*exp(-(abs(diff**2)/s))
        add_edge(i,j,i+1,j,w,g)
        if (j-1>=0): 
      
            diff=img[i][j]-img[i+1][j]
            w = k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i+1,j-1,w,g)
        if (j+1<img_size[0]):
     
            diff=img[i][j]-img[i+1][j+1]
            w = k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i+1,j+1,w,g)
    if (j-1>=0):
  
        diff=img[i][j]-img[i][j-1]
        w = k*exp(-(abs(diff**2)/s))
        add_edge(i,j,i,j-1,w,g)
    if (j+1<img_size[0]):

        diff=img[i][j]-img[i][j+1]
        w = k*exp(-(abs(diff**2)/s))
        add_edge(i,j,i,j+1,w,g)

def get_pos_from_node_number(node_number):
	row=math.floor( float(node_number)/float(img_size[0]) )
	col=node_number%img_size[0]
	return (int(row),col)

def make_graph(g):
    bar = progressbar.ProgressBar(maxval=img_size[0], \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            add_neighborhood_edges(i,j,g)
            add_t_links(i,j,g)
        bar.update(i+1)
    bar.finish()


start = time.time()
print("\n-Making graph")
g = maxflow.Graph[int]()
nodes = g.add_nodes(img_size[0]*img_size[1])



make_graph(g)
end = time.time()
print("-Done in "+str(end-start))
start = time.time()
print("\n-Initialize flow")
flow = g.maxflow()
print("Maximum flow:", flow)
end = time.time()
print("-Done in "+str(end-start))
#---------------------------------Getting the Min-Cut ------------------------------------#
cuts = g.get_grid_segments(nodes)
#print cuts

print("\n-Finding cut:")

start = time.time()
#forming new foreground image
Iout = ones(shape = nodes.shape)


mask=np.zeros((img_size[0],img_size[1],3))
for i in range(len(nodes)):
    Iout[i] = g.get_segment(nodes[i]) # calssifying each pixel as either forground or background
    if (Iout[i]==True):
        x,y=get_pos_from_node_number(i)
        img[x,y]=20

end = time.time()
print("-Done in "+str(end-start))
cv2.imwrite('foreground.png',img)


plt.imshow(img,vmin=0,vmax=255) # plot the output image
plt.show()
#cv2.imshow("output",img)
#cv2.waitKey(0)
