import cv2
import support
from ultralytics import YOLO

video_path = r"D:\DATA_FOLDER\HOC_TAP\NLCN\DISTANCE\images\demo.mp4"
image_path = r"D:\DATA_FOLDER\HOC_TAP\NLCN\DISTANCE\images\ex_280.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 480))
model = YOLO(r"D:\DATA_FOLDER\HOC_TAP\NLCN\DISTANCE\model\best_model.pt")

######### define real height of license plate 
measured_distance  = 280 #in centimeters
long_license_height = 12 #in centimeters
short_license_height = 16.5 #in centimeters

v_width, v_height = support.license_height_plate_dect(img, model, show=False)

############################################# calculate focal length 
focal_length = support.folcalLength_cal(distance=measured_distance, r_height=short_license_height, v_height=v_height)

print("Focal: {}, v_height: {}, v_width: {}".format(focal_length, v_height, v_width))

cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture(r"D:\DATA_FOLDER\HOC_TAP\NLCN\DISTANCE\images\demo_video2.mp4")
##############################################
# Set the desired frame width and height
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while True:
    distance = 0
    v_height, v_width = 0, 0
    ret, img = cap.read()
    v_width, v_height = support.license_height_plate_dect(img, model, show=False)

    # if regconizing any license height calculate the distance
    if(v_height > 0):
        #### set the real height of license in long and short situation
        if(v_width/v_height > 4):
            r_height = long_license_height
        else:
            r_height = short_license_height
        # calculate distance
        distance = support.distance_est(focal_length=focal_length, r_height=r_height, v_height=v_height)
        print("Heigth in img: ", v_height)
        print("distance: ", distance)
        
    #show image with distance or no detection
    support.show_cam(img, distance)
    if cv2.waitKey(10) & 0xFF==27:
        break   
    
cap.release()
cv2.destroyAllWindows()



#### This code for testing result in image
# test_img  = cv2.imread(r"D:\DATA_FOLDER\HOC_TAP\NLCN\DISTANCE\images\ex_75.jpg")
# test_img = cv2.resize(test_img, (640, 480))
# v_width, v_height = support.license_height_plate_dect(test_img, model)
# distance = support.distance_est(focal_length=focal_length, r_height=short_license_height, v_height=v_height)
# print("distance: {}, v_height: {}, v_width: {}, focal: {}".format(distance, v_height, v_width, focal_length) )

