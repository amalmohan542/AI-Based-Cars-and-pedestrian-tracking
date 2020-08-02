import cv2

#Our Image
img_file="carimage.jpeg"

#Pre-trained Car-detector
classifier_file="car_detector.xml"

#Create OpenCV image
img=cv2.imread(img_file)

#Convert to black and white image inorder to use Haar Features
# and makes the Algo Fast Without Dealing With colour Data.
blackandwhite= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create a car Classifier
car_tracker=cv2.CascadeClassifier(classifier_file)












#Display the images with Cars Spotted
cv2.imshow("AI car Detector", blackandwhite)

#Dont Auto close window and waite for a key presse
cv2.waitKey()




print("Code Completed Sucessfully")