############## Object detection and real world XYZ position estimation #############################
# for this to work you must have 2 cameras positioned in the same y and z planes with known separation
# you must have a white background
# this is just an estimation, for enhanced accuray you can calibrate the cameras individually and then together using as a stereo pair. 

import cv2
import numpy as np

alpha = 60  #camera field of view FOV in deg
camera_distance = 80 # distance between cameras in mm

#ensure cameras positions are correct 
capL = cv2.VideoCapture(0,cv2.CAP_DSHOW) # left camera
capR = cv2.VideoCapture(1,cv2.CAP_DSHOW) # right camera

print('video running, press Ecs to quit')

while capL.isOpened() & capR.isOpened():

    retL,frameL = capL.read()
    retR,frameR = capR.read()

    if retL == False or retR == False:
        print('not able to read camera')
        break

    #some simple image processing to aid thresholding
    gray_imageL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    gray_imageR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    blurred_imageL = cv2.blur(gray_imageL, (3,3))
    blurred_imageR = cv2.blur(gray_imageR, (3,3))
    
    #apply thresholding to detect objects against the white background. value can be refined through trial and error
    threshold = 70 

    ret2L,threshL = cv2.threshold(blurred_imageL,threshold,255,cv2.THRESH_BINARY_INV)    
    ret2R,threshR = cv2.threshold(blurred_imageR,threshold,255,cv2.THRESH_BINARY_INV)

    #find contours in the image
    contoursL, hierarchyL = cv2.findContours(threshL, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contoursR, hierarchyR = cv2.findContours(threshR, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    if len(contoursL) > 0:
            
        if len(contoursL) == len(contoursR):
        
            #number of first object
            i=1
            j=1

            #min size of bounding box, this will prevent any small noise getting picked up as an object, refine as required. 
            min_size = 50

            #object center points which will be called later when doing the depth estimation
            center_pointsL = []
            center_pointsR = []
            text_pointsR = []

            for cntL in contoursL:

                #compute the bounding box of the contour
                xL,yL,wL,hL = cv2.boundingRect(cntL)

                #check if bounding box is above min size
                if wL >= min_size and hL >= min_size:

                    centerL = (int(xL+wL/2),int(yL+hL/2))

                    center_pointsL.append(centerL)

                    i += 1
                    
            for cntR in contoursR:
            
                #compute the bounding box of the contour
                xR,yR,wR,hR = cv2.boundingRect(cntR)

                #check if bounding box is above min size
                if wR >= min_size and hR >= min_size:

                    centerR = (int(xR+wR/2),int(yR+hR/2))
                    cv2.circle(frameR,centerR,5,(0,0,255),-1)
                    cv2.rectangle(frameR,(xR,yR),(xR+wR,yR+hR),(0,0,255),2)

                    center_pointsR.append(centerR)

                    #label object
                    text_x = int(xR)
                    text_y = int(yR + hR + 20)
                    cv2.putText(frameR, "Object" + str(j), (text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                    
                    #this passes the bounding box co-ordinates that we can use to add the position later in programme
                    text_pointsR.append((text_x,text_y))

                    j += 1

            if len(center_pointsL) > 0 and len(center_pointsL) == len(center_pointsR): 
                for i in range(len(center_pointsL)):

                    #forumlas taken from https://github.com/niconielsen32/ComputerVision/tree/master/StereoVision/Python 

                    height_right, width_right, depth_right = frameR.shape
                    f_pixel_x = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
                    f_pixel_y = (height_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

                    # CALCULATE THE DISPARITY:
                    x_right = center_pointsR[i][0]
                    x_left = center_pointsL[i][0]
                    disparity = x_left-x_right      #pixel displacement between left and right frames

                    ############ calculate X, Y, Z positions ###############
                    #scale positions based on trial and error using ruler to measure realworld position
                    #all measurement in mm
                    #based on https://staff.fnwi.uva.nl/a.visser/education/bachelorAI/Honours_Extension_Niels_Sombekke.pdf

                    xPos = (camera_distance * (center_pointsR[i][0] - width_right/2)) / disparity 
                    xPos = int(xPos)

                    yPos = (camera_distance * f_pixel_x * (center_pointsR[i][1] - height_right/2)) / (f_pixel_y * disparity) # 0 value is the y component of the principle point
                    yPos = int(yPos)

                    zDepth = abs((camera_distance*f_pixel_x)/disparity)
                    zDepth = abs(int(zDepth * 1.4))

                    cv2.putText(frameR,f'x: {xPos}, y: {yPos}, z: {zDepth}',(text_pointsR[i][0],text_pointsR[i][1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            
            else: 
                cv2.putText(frameR,'depth detection failed',(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        else:
            cv2.putText(frameR,'object not visible to both cameras',(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    else: 
        cv2.putText(frameR,'no object deteced',(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    cv2.imshow('XYZ Object Dectection',frameR)
    
    if cv2.waitKey(1) == 27:
        print('video stopped')
        break
        
capL.release()
capR.release()
cv2.destroyAllWindows()