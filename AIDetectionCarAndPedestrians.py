import cv2

#Our Image
img_file="carimage.jpeg"
#Out Video footage to check
Video=cv2.VideoCapture("CarsDetectVideo.mkv")

#Pre-trained Car-Classifier
classifier_file="car_detector.xml"

#Create a car Classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

# Used to run for ever until the car stops off.
while True:
    #Read the current Frame from video
    #Return 2 values whether the raed is sucessful or not and frame
    (read_sucessful, frame) = Video.read() 
    #To check whether raed is sucessful or not
    if read_sucessful:
        #Convert to black and white image inorder to use Haar Features
        # and makes the Algo Fast Without Dealing With colour Data.
        grayscaledframe= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    

    #Car Tracker (Detect Cars of Multile Size or Scale)
    cars=car_tracker.detectMultiScale(grayscaledframe)

    #Draw Rectangles around Cars Detected
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    print(cars)


    

    #Display the images with Cars Spotted
    cv2.imshow("AI car Detector", frame)


    #Dont Auto close window and waite for a key presse
    cv2.waitKey(1)



"""

#Create OpenCV image
img=cv2.imread(img_file)

#Convert to black and white image inorder to use Haar Features
# and makes the Algo Fast Without Dealing With colour Data.
blackandwhite= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create a car Classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

#Car Tracker (Detect Cars of Multile Size or Scale)
cars=car_tracker.detectMultiScale(blackandwhite)

#Draw Rectangles around Cars Detected
car1=cars[3]
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)





#Display the images with Cars Spotted
cv2.imshow("AI car Detector", img)


#Dont Auto close window and waite for a key presse
cv2.waitKey()

"""


print("Code Completed Sucessfully")