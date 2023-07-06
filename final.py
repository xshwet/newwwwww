#!/home/einfochips/Desktop/YOLO/evn/bin/python3

import torch
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Int16
from cv_bridge import CvBridge,CvBridgeError
import time
import math
from tesingm.srv import servicetest
from tesingm.srv import servicetestRequest
from tesingm.srv import servicetestResponse
from threading import Thread


moving = False
model = torch.hub.load('ultralytics/yolov5','custom',path='/home/einfochips/Desktop/YOLO/yolov5/runs/train/exp4/weights/last.pt')
# model.conf = 0.65
model.classes = (0,1,2,3,5,6,7)
a = model.classes
steps_name = {
   0: "ScreenOff",
   1: "ScreenOn",
   2: "Zero",
   3: "Settings",
   5: "Bluetooth",
   6: "BtnOff",
   7: "BtnOn",
}

K=np.array([[434.2196960449219, 0.0, 266.6663412990292],[0.0, 491.9661865234375,238.71625361069528],[0.0, 0.0, 1.0]],dtype=np.float64).reshape(3,3)
R=np.array([[-0.050112674474709636], [-3.0323416167631896],[0.012589481083534122]],dtype=np.float64).reshape(3,1)
T=np.array([[16.512429040858382], [51.675548988183714], [202.46864073880207]],dtype=np.float64).reshape(3,1)
cur_x, cur_y, cur_z = 0, -163,212

steps_count = (0,1,2,3,5,6,7)
steps = 0


def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invR
        worldPtPlane = np.dot(inv_r, worldPtCam)  # 3*3 dot 3*1 = 3*1
        # zc
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T213596990823
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt

def show(image):

    global results

    bridge = CvBridge()
    open=bridge.imgmsg_to_cv2(image)
    col=cv2.cvtColor(open,cv2.COLOR_BGR2RGB)	
    results = model(col)
   
    cv2.imshow('YOLO',np.squeeze(results.render()))
    cv2.waitKey(1)
    


def detect():
    try:
        if len(torch.tensor(results.xyxy[0])) <= 1 and int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps]:
            c_x = ((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2)
            c_y = ((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)

            x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
            
            t = math.sqrt(x * x + y * y + 120 * 120) / 120 
            
            new_x, new_y = cur_x + x, cur_y + y + 15
            nn_new_x = new_x - 25
            #print("continuous xandy for:",steps_count[steps]  ,nn_new_x, new_y)
            return (str(nn_new_x) +" "+ str(new_y))
        
        elif len(torch.tensor(results.xyxy[0])) > 1 and int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps]:
                c_x = ((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2)
                c_y = ((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)
                
                x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
                
                t = math.sqrt(x * x + y * y + 120 * 120) / 120 
               
                new_x, new_y = cur_x + x, cur_y + y + 15
                nn_new_x = new_x - 25
                #print("continuous xandy for:",steps_count[steps]  ,nn_new_x, new_y)
                return (str(nn_new_x) +" "+ str(new_y))
        
        elif len(torch.tensor(results.xyxy[0])) > 1 and int(torch.tensor(results.xyxy[0])[1][5].item()) == steps_count[steps]:
                c_x = ((int(torch.tensor(results.xyxy[0])[1][0].item())+int(torch.tensor(results.xyxy[0])[1][2].item()))//2)
                c_y = ((int(torch.tensor(results.xyxy[0])[1][1].item())+int(torch.tensor(results.xyxy[0])[1][3].item()))//2)
                
                x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
                
                t = math.sqrt(x * x + y * y + 120 * 120) / 120 
               
                new_x, new_y = cur_x + x, cur_y + y + 15
                nn_new_x = new_x - 25
                #print("continuous xandy for:",steps_count[steps]  ,nn_new_x, new_y)
                return (str(nn_new_x) +" "+ str(new_y))
        else:
            pass
    except IndexError:
            pass
    

def requesting():
    global client
    send_coord = detect()
    # print("got coordinates:", send_coord)
    # print("steps:", steps)
    if send_coord:
        try:
            client = client_obj(send_coord)
            return client.reply
        except:
            print("Service call failed")
   







def main():
    cam_sub = rospy.Subscriber("/usb_cam/image_rect_color",Image,show)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    









if __name__=="__main__":
    rospy.init_node("opencv")
    client_obj = rospy.ServiceProxy('test_service', servicetest)
    t1 = Thread(target=main)
    t1.start()
    rospy.wait_for_service('test_service')
    time.sleep(1)
    try:
        while True:
            s = requesting()
            # print("got s:", s)
            # print("global steps:",steps)
            if s == "Done" or len(torch.tensor(results.xyxy[0])) >= 1:
                try:
                    #print("current class detected is :",int(torch.tensor(results.xyxy[0])[0][5].item()))
                    if int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps + 1] or int(torch.tensor(results.xyxy[0])[1][5].item()) == steps_count[steps + 1]:
                        steps += 1
                        # print("Global Steps incremented to:-", steps)
                        if steps>6:
                            break

                    elif int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps - 1] or int(torch.tensor(results.xyxy[0])[1][5].item()) == steps_count[steps - 1] :
                        steps -= 1
                        # print("Global Steps decremented to:-", steps)
                        if steps>6:
                            break
                        
                    else:
                        # print("Global Steps not incrementing:-", steps)
                        pass
                except: 
                    pass
            else:
                pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

