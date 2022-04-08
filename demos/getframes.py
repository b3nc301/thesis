import cv2
import random
import os


videoLink = "IMG_2567.MP4"
# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(videoLink)
i = 0
j=0
rand = int(random.random()*100)
os.mkdir(videoLink+"images")
while(cap.isOpened()):
    ret, frame = cap.read()
     
    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
    if j == 15:
        break
    if(i==rand):
        print(i)
        cv2.imwrite(videoLink+'images/Frame'+str(i)+'.jpg', frame)
        rand = int(random.random()*100)+i
        j += 1
    # Save Frame by Frame into disk using imwrite method
    i += 1
 
cap.release()
cv2.destroyAllWindows()