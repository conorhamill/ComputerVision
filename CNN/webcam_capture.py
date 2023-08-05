import cv2
import os

#create folder for item name given by below - change as required
#change this to function if required
item = 'test'
directory = os.getcwd()
path = os.path.join(directory,'items',item)
os.mkdir(path)

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

#first name of image
i=0

print('Video starting, press Esc Key to quit')

while cap.isOpened():

    ret,frame = cap.read()

    if ret != True:
        print('Video capture failed')
        break

    cv2.imshow('Webcam', frame)

    k =  cv2.waitKey(1)

    #on enter keypress save an image to the folder and increment the filename
    if k == 32:
        
        file_name = str(i) + '.jpg'
        cv2.imwrite(os.path.join(path,file_name),frame)
        i += 1

        print('Image captured')

    elif k == 27:
        print('Video stopped')
        print(f'you captured {i} images')
        break 


cap.release()
cv2.destroyAllWindows()
