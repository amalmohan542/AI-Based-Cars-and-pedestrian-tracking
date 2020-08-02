import cv2

#Our Image
img_file="carimage.jpeg"

#Pre-trained Car-detector
classifier_file="car_detector.xml"

#Create OpenCV image
img=cv2.imread(img_file)

#Display the images with Cars Spotted
cv2.imshow("AI car Detector", img)

#Dont Auto close window and waite for a key presse
cv2.waitKey()




print("Code Completed Sucessfully")