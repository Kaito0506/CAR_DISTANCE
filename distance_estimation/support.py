import cv2
from ultralytics import YOLO


# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
  
# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

def license_height_plate_dect(img, model, show=False):
    width = 0 
    height = 0
    result = model(img, show=False)
    boxes = result[0].boxes
    for box in boxes:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bot_right_x = int(box.xyxy.tolist()[0][2])
        bot_right_y = int(box.xyxy.tolist()[0][3])
        height = bot_right_y - top_left_y
        width = bot_right_x - top_left_x
        cv2.rectangle(img, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0,   255), 2)    
        if show:
            cv2.imshow("Frame", img)
            if cv2.waitKey(100) & 0xFF==27:
                break   
    return width, height
    
# function for calculating focal length of camera: v_height/facolength = r_height/distance
def folcalLength_cal(distance, r_height, v_height):
    focal =  (v_height*distance)/(r_height)
    return focal

# function for distance estimating from license plate 
def distance_est(focal_length, r_height, v_height):
    distance = (focal_length*r_height)/ (v_height)
    return distance


def convert(distance_in_centimeters):
    distance_in_meters = distance_in_centimeters/100
    return distance_in_meters

def show_cam(img, distance):
    distance = convert(distance)
    if(distance > 0):
        cv2.putText(img=img,text="distance: {:.2f} m".format(distance) , org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)
    else:
       cv2.putText(img=img,text="no detection" , org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)
    
    if distance!=0 and distance <=1.5:
        cv2.putText(img=img,text="DANGER!!!" , org=(50, 150), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=RED, thickness=3)
    cv2.imshow("Frame", img)
    
    if cv2.waitKey(10) & 0xFF==27:
        cv2.destroyAllWindows()
    
    
    

##################################################################
# v_height = license_height_plate_dect(img, model)
# focal = folcalLength_cal(distance, r_height, v_height)





