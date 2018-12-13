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



img = cv2.imread('rose.jpg', cv2.IMREAD_GRAYSCALE)
#img_size =(len(img),len(img[0]))
img_size =(50,50)
#img.resize((50,50))
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



print("\n-Calculating foreground/background probability for each pixel")
#start calculating foreground/background probability for each 
start=time.time()
for i in range(len(img)):
    for j in range(len(img[0])):
        #predict_result_fore = round(gmm.predict_proba([ [img[i,j]] ])[0][0] , 5)+0.000001   #added 0.00001 to prevent divide by zero
        #predict_result_back = round(gmm.predict_proba([ [img[i,j]] ])[0][1] , 5)+0.000001	#added 0.00001 to prevent divide by zero
        #Rp[i][j]=math.log(K*predict_result_fore/predict_result_back) 
        PF[i][j]=round( gmm.predict_proba([ [img[i,j]] ])[0][0] ,5)
        PB[i][j]=round( gmm.predict_proba([ [img[i,j]] ])[0][1] ,5)


#print Rp
end=time.time()
print("-Done in "+str(end-start))




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
    print " normal edge: "+ str(weight)

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
     #   w = k*exp(-(30**2)/10)
        #add_edge(i-1,j,i,j,w,g)
        add_edge(i,j,i-1,j,w,g)
        if (j-1>=0): 
        #    add_edge(i-1,j-1,i,j,w,g)
            diff = img[i][j]-img[i-1][j-1]
            w =  k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i-1,j-1,w,g)
        if (j+1<img_size[0]):
        #    add_edge(i-1,j+1,i,j,w,g)
            diff = img[i][j]-img[i-1][j+1]
            w = k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i-1,j+1,w,g)
    if (i+1<img_size[0]):
    #    add_edge(i+1,j,i,j,w,g)
        diff = img[i][j]-img[i+1][j]
        w = k*exp(-(abs(diff**2)/s))
        add_edge(i,j,i+1,j,w,g)
        if (j-1>=0): 
        #    add_edge(i+1,j-1,i,j,w,g)
            diff=img[i][j]-img[i+1][j]
            w = k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i+1,j-1,w,g)
        if (j+1<img_size[0]):
        #    add_edge(i+1,j+1,i,j,w,g)
            diff=img[i][j]-img[i+1][j+1]
            w = k*exp(-(abs(diff**2)/s))
            add_edge(i,j,i+1,j+1,w,g)
    if (j-1>=0):
    #    add_edge(i,j-1,i,j,w,g)
        diff=img[i][j]-img[i][j-1]
        w = k*exp(-(abs(diff**2)/s))
        add_edge(i,j,i,j-1,w,g)
    if (j+1<img_size[0]):
    #    add_edge(i,j+1,i,j,w,g)
        diff=img[i][j]-img[i][j+1]
        w = k*exp(-(abs(diff**2)/s))
        add_edge(i,j,i,j+1,w,g)

def get_pos_from_node_number(node_number):
	row=math.floor( float(node_number)/float(img_size[0]) )
	col=node_number%img_size[0]
	return (int(row),col)

def make_graph(g):
    for i in range(img_size[0]):
        for j in range(img_size[0]):
            add_neighborhood_edges(i,j,g)
            add_t_links(i,j,g)


start = time.time()
print("\n-Making graph")
g = maxflow.Graph[int]()
nodes = g.add_nodes(img_size[0]*img_size[1])

#pic=maxflow.Graph[int]()
#nodeids=pic.add_grid_nodes(img_size) # Adding non-nodes
#pic.add_grid_edges(nodeids,0),pic.add_grid_tedges(nodeids, img, 255-img)
#gr = pic.maxflow()
#IOut = pic.get_grid_segments(nodeids)

make_graph(g)

end = time.time()
print("-Done in "+str(end-start))

start = time.time()
print("\n-Initialize flow")
flow = g.maxflow()
print "Maximum flow:", flow
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

print Iout

end = time.time()
print("-Done in "+str(end-start))
cv2.imwrite('foreground.png',img)


plt.imshow(img,vmin=0,vmax=255) # plot the output image
plt.show()
#cv2.imshow("output",img)
#cv2.waitKey(0)
