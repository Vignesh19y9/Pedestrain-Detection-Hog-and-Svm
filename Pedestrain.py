import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from skimage import color
import matplotlib.pyplot as plt 
import os 





def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])
    
    


clf=joblib.load('human.npy')

im = cv2.imread('2.jpg')

im = imutils.resize(im, width = min(400, im.shape[1]))
cv2.imshow('Input',im)

min_wdw_sz = (64, 128)
step_size = (15, 15)
downscale = 1.25

detections = []
    #The current scale of the image 


scale = 0

for im_scaled in pyramid_gaussian(im, downscale = downscale):
    cd=[]
    #The list contains detections at the current scale
    if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
        break
    for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        im_window = color.rgb2gray(im_window)
        fd = hog(im_window, orientations=9, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False)

        fd = fd.reshape(1, -1)
        pred = clf.predict(fd)

        if pred == 1:
            
            if clf.decision_function(fd) > 0.5:
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                int(min_wdw_sz[0] * (downscale**scale)),
                int(min_wdw_sz[1] * (downscale**scale))))
                cd.append(detections[-1])
        clone = im_scaled.copy()
              
        for x1, y1, _, _, _  in cd:             
            # Draw the detections at this scale
            cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                im_window.shape[0]), (0, 0, 0), thickness=2)
        cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
            im_window.shape[0]), (255, 255, 255), thickness=2)
        cv2.imshow("Sliding Window in Progress", clone)
        cv2.waitKey(30)

        
    scale += 1

clone = im.copy()

for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
print( "sc: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
print ("shape, ", pick.shape)

for(xA, yA, xB, yB) in pick:
    cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

'''plt.axis("off")
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title("Raw Detection before NMS")
plt.show()'''
cv2.imshow('Raw Detection before Non Maximum Suppression',im)
cv2.waitKey(0)

'''plt.axis("off")
plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
plt.title("Final Detections after applying NMS")
plt.show()'''
cv2.imshow('Final Detections after Non Maximum Suppression',clone)

cv2.waitKey(0)
