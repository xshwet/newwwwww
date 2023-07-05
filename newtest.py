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
import threading



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

steps_count = (0,1,2,3,5,6,7)
steps = 0
nn_new_x= None
new_y = None
old_coord = [0.0, -163.0]
new_steps = 0
c_x = None
c_y = None







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

def detect(image):
    
    global steps_count, steps, nn_new_x, new_y, results

    bridge = CvBridge()
    open=bridge.imgmsg_to_cv2(image)
    col=cv2.cvtColor(open,cv2.COLOR_BGR2RGB)	
    results = model(col)
   
    cv2.imshow('YOLO',np.squeeze(results.render()))
    cv2.waitKey(1)
    # a = torch.tensor(results.xyxy[0])[0][0].item()
    # print(a[0][0].item())
    # print(int(torch.tensor(results.xyxy[0])[0][5].item()))





    try:
        if int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps] and len(torch.tensor(results.xyxy[0])) == 1:
            if a[steps] == steps_count[steps]:
                # cv2.circle(img=col,center=(((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2),((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)),radius=3,color=(0,255,0),thickness=-1)
                #cv2.imshow('YOLO',np.squeeze(results.render()))
                # rate = rospy.Rate(0.5)
                c_x = ((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2)
                c_y = ((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)
                # print("center points of object == :", c_x,c_y)
                #zp = int(torch.tensor(results.xyxy[0])[0][5].item())
                cur_x, cur_y, cur_z = 0, -163,212
                K=np.array([[434.2196960449219, 0.0, 266.6663412990292],[0.0, 491.9661865234375,238.71625361069528],[0.0, 0.0, 1.0]],dtype=np.float64).reshape(3,3)
                R=np.array([[-0.050112674474709636], [-3.0323416167631896],[0.012589481083534122]],dtype=np.float64).reshape(3,1)
                T=np.array([[16.512429040858382], [51.675548988183714], [202.46864073880207]],dtype=np.float64).reshape(3,1)
               
                x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
                
                t = math.sqrt(x * x + y * y + 120 * 120) / 120 
               
                new_x, new_y = cur_x + x, cur_y + y + 15
                nn_new_x = new_x - 25
                # print("continuous xandy for:",steps_count[steps]  ,nn_new_x, new_y)
                    #jetmax.set_position((nn_new_x, new_y, 80), t)
                    #rospy.sleep(t)
                    #jetmax.set_position((new_x , new_y, 70), 0.3)
                    #rospy.sleep(t + 0.6)
                   
                
            else:
                pass
        elif int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps] and len(torch.tensor(results.xyxy[0])) > 1:
                c_x = ((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2)
                c_y = ((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)
                # print("center points of object 0>1 :", c_x,c_y)
                #zp = int(torch.tensor(results.xyxy[0])[0][5].item())
                cur_x, cur_y, cur_z = 0, -163,212
                K=np.array([[434.2196960449219, 0.0, 266.6663412990292],[0.0, 491.9661865234375,238.71625361069528],[0.0, 0.0, 1.0]],dtype=np.float64).reshape(3,3)
                R=np.array([[-0.050112674474709636], [-3.0323416167631896],[0.012589481083534122]],dtype=np.float64).reshape(3,1)
                T=np.array([[16.512429040858382], [51.675548988183714], [202.46864073880207]],dtype=np.float64).reshape(3,1)
               
                x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
                
                t = math.sqrt(x * x + y * y + 120 * 120) / 120 
               
                new_x, new_y = cur_x + x, cur_y + y + 15
                nn_new_x = new_x - 25
                # print("continuous xandy for:",steps_count[steps]  ,nn_new_x, new_y)
        
        elif int(torch.tensor(results.xyxy[0])[1][5].item()) == steps_count[steps] and len(torch.tensor(results.xyxy[0])) > 1:
                c_x = ((int(torch.tensor(results.xyxy[0])[1][0].item())+int(torch.tensor(results.xyxy[0])[1][2].item()))//2)
                c_y = ((int(torch.tensor(results.xyxy[0])[1][1].item())+int(torch.tensor(results.xyxy[0])[1][3].item()))//2)
                # print("center points of object 1>1 :", c_x,c_y)
                #zp = int(torch.tensor(results.xyxy[0])[0][5].item())
                cur_x, cur_y, cur_z = 0, -163,212
                K=np.array([[434.2196960449219, 0.0, 266.6663412990292],[0.0, 491.9661865234375,238.71625361069528],[0.0, 0.0, 1.0]],dtype=np.float64).reshape(3,3)
                R=np.array([[-0.050112674474709636], [-3.0323416167631896],[0.012589481083534122]],dtype=np.float64).reshape(3,1)
                T=np.array([[16.512429040858382], [51.675548988183714], [202.46864073880207]],dtype=np.float64).reshape(3,1)
               
                x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
                
                t = math.sqrt(x * x + y * y + 120 * 120) / 120 
               
                new_x, new_y = cur_x + x, cur_y + y + 15
                nn_new_x = new_x - 25
                # print("continuous xandy for:",steps_count[steps]  ,nn_new_x, new_y)

        else:
            # nn_new_x= None
            # new_y = None
            # c_x = None
            # c_y = None
            pass
    except IndexError:
            cv2.imshow('YOLO',col)
            cv2.waitKey(1)
    #cv2.imshow('YOLO',np.squeeze(results.render()))
    cv2.imshow('YOLO',col)
    
    cv2.waitKey(1)


def requesting(msg):
    global old_coord, new_steps, results
   
    # if nn_new_x == old_coord[0] and new_y == old_coord[1] and new_steps == steps and len(torch.tensor(results.xyxy[0])) >= 1:
    #     print("old coord:", old_coord)
    #     time.sleep(0.5)
    #     client = client_obj((str(nn_new_x) +" "+ str(new_y)))
    #     return client.reply
    # else:
    #     new_steps = steps
    #     old_coord = [nn_new_x,new_y]
    #     new_steps = steps
    #     print("changing coord:", old_coord)
    
    try:
        time.sleep(0.5)
        print(str(nn_new_x) +" "+ str(new_y))
        client = client_obj((str(nn_new_x) +" "+ str(new_y)))
        return client.reply
    except:
        print("Service call failed")


   


def main():
    
    cam_sub = rospy.Subscriber("/usb_cam/image_rect_color",Image,detect)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    

if __name__=="__main__":
    rospy.init_node("opencv")
    client_obj = rospy.ServiceProxy('test_service', servicetest)
    lock = threading.RLock()
    t1 = threading.Thread(target = main)
    t1.start()    
    #print("moving")
    #time.sleep(2)
    while True:
        time.sleep(3)
        # print("Global Steps:-", steps)
        # print("global x&y:",nn_new_x, new_y)
        # print("midpoints x&y:",c_x,c_y)
        rospy.wait_for_service('test_service')
        # if nn_new_x == None:
        #     print("pass")
        # else:
        #     #time.sleep(1)
            # print("latest coord:", nn_new_x, new_y)
            
        with lock:
            c = str(nn_new_x)+ str(new_y)
            print("c",c)
            s = requesting(c)
            # print("Reply:",s)
            if s == "Done":
                try:
                    while True:
                        if int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps]:
                            if steps>6:
                                break
                            steps += 1
                            # print("Global Steps after incrementing:-", steps)
                            # print("next to detect:" , steps_name[steps])
                            break
                        else:
                            # print("Global Steps remained same:-", steps)
                            break
                except: 
                    pass
            elif s == "None":
                pass
