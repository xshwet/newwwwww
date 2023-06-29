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
import threading

pub = rospy.Publisher("/coordinates",Point,queue_size=10)
model = torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp4/weights/last.pt')
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
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


# def newa(coord):
#     rate = rospy.Rate(1)
#     pub.publish(coord)
#     rate.sleep()






















def detect(image):
    
    global steps_count, steps, results, col, a

    bridge = CvBridge()
    open=bridge.imgmsg_to_cv2(image)
    col=cv2.cvtColor(open,cv2.COLOR_BGR2RGB)	
    results = model(col)
   
    # cv2.imshow('YOLO',np.squeeze(results.render()))
    # cv2.waitKey(1)
    # a = torch.tensor(results.xyxy[0])[0][0].item()
    # print(a[0][0].item())
    # print(int(torch.tensor(results.xyxy[0])[0][5].item()))





    try:
        if int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps]:
            if a[steps] == steps_count[steps]:
                # cv2.circle(img=col,center=(((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2),((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)),radius=3,color=(0,255,0),thickness=-1)
                cv2.imshow('YOLO',np.squeeze(results.render()))
                # rate = rospy.Rate(0.5)
                global coord
                coord = Point()
                c_x = ((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2)
                c_y = ((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)
                zp = int(torch.tensor(results.xyxy[0])[0][5].item())
                cur_x, cur_y, cur_z = 0, -163,212 # 吸嘴的当前坐标
                K=np.array([[434.2196960449219, 0.0, 266.6663412990292],[0.0, 491.9661865234375,238.71625361069528],[0.0, 0.0, 1.0]],dtype=np.float64).reshape(3,3)
                R=np.array([[-0.050112674474709636], [-3.0323416167631896],[0.012589481083534122]],dtype=np.float64).reshape(3,1)
                T=np.array([[16.512429040858382], [51.675548988183714], [202.46864073880207]],dtype=np.float64).reshape(3,1)
                # 计算卡片在现实世界的中吸嘴中心(外参标定的时候用吸嘴标定的)的坐标
                x, y, _ = camera_to_world(K,R,T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0]
                pub = rospy.Publisher("/coordinates",Point, queue_size=1)
                t = math.sqrt(x * x + y * y + 120 * 120) / 120 # 计算卡片位置的距离, 通过距离/速度=时间, 计算用多少时间到达卡片位置
                new_x, new_y = cur_x + x, cur_y + y + 15 # 计算卡片位置相对于机械臂基座的坐标
                nn_new_x = new_x - 25
                    # 机械臂分步运行到卡片位置
                    #jetmax.set_position((nn_new_x, new_y, 80), t)
                    #rospy.sleep(t)
                    #jetmax.set_position((new_x , new_y, 70), 0.3)
                    #rospy.sleep(t + 0.6)
                coord.x = nn_new_x
                coord.y = new_y
                coord.z = zp
                # rate = rospy.Rate(0.2)

                lock = threading.Lock()
                with lock:
                    process(coord)
                time.sleep(1)

                # steps += 1
                # if steps < 7:
                #     print("next to detect", steps_name[steps_count[steps]])
            
                
                # steps += 1
                # rate.sleep()
                
            else:
                pass
        else:
            pass
    except IndexError:
            cv2.imshow('YOLO',col)
            cv2.waitKey(1)
    #cv2.imshow('YOLO',np.squeeze(results.render()))
    cv2.imshow('YOLO',col)
    
    cv2.waitKey(1)


def process(coord):    
    global steps_count, steps, results, col, a
    rate = rospy.Rate(1)
    print("publisher",coord)
    pub.publish(coord)
    rate.sleep()
    steps += 1
    if steps < 7:
        print("next to detect", steps_name[steps_count[steps]])
    else:
        pass



# def a():
#     global steps_count, steps, results, col
#     try:
#         if int(torch.tensor(results.xyxy[0])[0][5].item()) == steps_count[steps]:
#             if a[steps] == steps_count[steps]:
#                 cv2.circle(img=col,center=(((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2),((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)),radius=3,color=(0,255,0),thickness=-1)
#                 cv2.imshow('YOLO',np.squeeze(results.render()))
#                 rate = rospy.Rate(2)
#                 coord = Point()
#                 coord.x = ((int(torch.tensor(results.xyxy[0])[0][0].item())+int(torch.tensor(results.xyxy[0])[0][2].item()))//2)
#                 coord.y = ((int(torch.tensor(results.xyxy[0])[0][1].item())+int(torch.tensor(results.xyxy[0])[0][3].item()))//2)
#                 coord.z = int(torch.tensor(results.xyxy[0])[0][5].item())
#                 pub.publish(coord)
#                 steps += 1
#                 # rate.sleep()
                
#             else:
#                 pass
#         else: 
#             cv2.imshow('YOLO',np.squeeze(results.render()))
#     except IndexError:
#             pass
#     #cv2.imshow('YOLO',np.squeeze(results.render()))
#     cv2.imshow('frame',col)
    
#     cv2.waitKey(1)



   


def main():
    rospy.init_node("opencv")
    cam_sub = rospy.Subscriber("/usb_cam/image_rect_color",Image,detect)
    # feedback_sub = rospy.Subscriber("/feedbackstring",Int16,process,queue_size=1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    

if __name__=="__main__":
    main()
