import cv2

#Our Video footage to Detect cars
Video=cv2.VideoCapture("CarsDetectVideo.mkv")
#Our Video footage to Detect cars and pedestrians
# Video=cv2.VideoCapture("Pedestrians.mkv")

#Pre-trained Car-Classifier and Pedestrians-Classifier file
car_tracker_file="car_detector.xml"
Pedestrian_tracker_file="pedestrianDetector.xml"

#Create a car Classifier and pedestrian Classifier
car_tracker=cv2.CascadeClassifier(car_tracker_file)
Pedestrian_tracker=cv2.CascadeClassifier(Pedestrian_tracker_file)



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
    

    #Detect Cars and Pedestrians (of Multile Size or Scale)
    cars=car_tracker.detectMultiScale(grayscaledframe)
    Pedestrians=Pedestrian_tracker.detectMultiScale(grayscaledframe)

    #Draw Rectangles around Cars Detected
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    #Draw Rectangles around Pedestrians Detected
    for (x,y,w,h) in Pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    
    #Display the images with Cars and Predestrians Spotted
    cv2.imshow("AI car and Pedestrians Detector", frame)


    #Dont Auto close window and waite for a key presse
    key=cv2.waitKey(1)

    #Press Q to Stop Running program
    if key==81 or key==113:
        break
#Release Video Capture File Object
Video.release()

print("Code Completed Sucessfully")