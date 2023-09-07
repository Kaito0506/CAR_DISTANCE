import cv2
import support
from ultralytics import YOLO

video_path = r"E:\HOC_TAP\NLCN\DISTANCE\images\demo.mp4"
image_path = r"E:\HOC_TAP\NLCN\DISTANCE\images\sample_200cm.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 480))
model = YOLO(r"E:\HOC_TAP\NLCN\DISTANCE\model\license_detector1.pt")

######### define real height of license plate 
measured_distance  = 200 #in centimeters
long_license_height = 12 #in centimeters
short_license_height = 17 #in centimeters

v_width, v_height = support.license_height_plate_dect(img, model, show=False)
############################################# calculate focal length of 2 type of license plate
long_focal = support.folcalLength_cal(distance=measured_distance, r_height=long_license_height, v_height=v_height)
short_focal = support.folcalLength_cal(distance=measured_distance, r_height=short_license_height, v_height=v_height)

print(long_focal, short_focal, v_height)

cap = cv2.VideoCapture(1)
##############################################
# Set the desired frame width and height
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while True:
    ret, img = cap.read()
    v_width, license_height_in_img = support.license_height_plate_dect(img, model, show=False)
    distance = 0
    # if regconizing any license height calculate the distance
    if(license_height_in_img > 0):
        distance = support.distance_est(focal_length=short_focal, r_height=short_license_height, v_height=license_height_in_img)
        print("Heigth in img: ", license_height_in_img)
        print("distance: ", distance)
        
    #show image with distance or no detection
    support.show_cam(img, distance)
    if cv2.waitKey(10) & 0xFF==27:
        break   
    
cap.release()
cv2.destroyAllWindows()

# test_img  = cv2.imread(r"E:\HOC_TAP\NLCN\DISTANCE\images\sample_150cm.jpg")
# test_img = cv2.resize(test_img, (640, 480))
# width, height = support.license_height_plate_dect(test_img, model)
# distance = support.distance_est(focal_length=short_focal, r_height=short_license_height, v_height=height)
# print("distance: {} , v_height: {}".format(distance, height) )

