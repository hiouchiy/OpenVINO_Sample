#import debug

import sys
import math
import random

import numpy as np
from numpy import linalg as LA

import cv2
from scipy.spatial import distance
from openvino.inference_engine import IENetwork, IECore


device='CPU'

if device == 'CPU':
  precision='FP32'
else:
  precision='FP16'

cpu_ext = "C:\\tmp\\cpu_extension.dll" #Absolute Path for CPU Extension lib is needed
model_base_path= 'C:\\tmp\\Transportation' #Absolute Path for downloaded pre-trained model root folder is needed
model_det      = model_base_path+'\\object_detection\\face\\pruned_mobilenet_reduced_ssd_shared_weights\\dldt\\FP32\\face-detection-adas-0001'
model_hp       = model_base_path+'\\object_attributes\\headpose\\vanilla_cnn\\dldt\\FP32\\head-pose-estimation-adas-0001'
model_gaze     = model_base_path+'\\object_attributes\\gaze\\custom-architecture\\dldt\\FP32\\gaze-estimation-adas-0002'
model_landmark = model_base_path+'\\object_attributes\\facial_landmarks\\custom-35-facial-landmarks\\dldt\\FP32\\facial-landmarks-35-adas-0002'

_N = 0
_C = 1
_H = 2
_W = 3


def line(p1, p2):
  A = (p1[1] - p2[1])
  B = (p2[0] - p1[0])
  C = (p1[0]*p2[1] - p2[0]*p1[1])
  return A, B, -C

def intersection(L1, L2):
  D  = L1[0] * L2[1] - L1[1] * L2[0]
  Dx = L1[2] * L2[1] - L1[1] * L2[2]
  Dy = L1[0] * L2[2] - L1[2] * L2[0]
  x = Dx / D
  y = Dy / D
  return x,y

def intersection_check(p1, p2, p3, p4):
  tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
  tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
  td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
  td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
  return tc1*tc2<0 and td1*td2<0






def draw_gaze_line(img, coord1, coord2, laser_flag):
  if laser_flag == False:
    # simple line
    cv2.line(img, coord1, coord2, (0, 0, 255),2)
  else:
    # Laser mode :-)
    beam_img = np.zeros(img.shape, np.uint8)
    for t in range(20)[::-2]:
      cv2.line(beam_img, coord1, coord2, (0, 0, 255-t*10), t*2)
    img |= beam_img


def draw_spark(img, coord):
  for i in range(20):
    angle = random.random()*2*math.pi
    dia   = random.randrange(10,60)
    x = coord[0] + int(math.cos(angle)*dia - math.sin(angle)*dia)
    y = coord[1] + int(math.sin(angle)*dia + math.cos(angle)*dia)
    cv2.line(img, coord, (x, y), (0, 255, 255), 2)



def usage():
  print("""
Gaze estimation demo
'f': Flip image
'l': Laser mode on/off
's': Spark mode on/off
'b': Boundary box on/off
""")


def main():

    usage()

    boundary_box_flag = True

    # Prep for face detection
    ie_det = IECore()
    if device == 'CPU':
      ie_det.add_extension(cpu_ext, 'CPU')    # absolute path is required
      #ie_det.add_extension('C:/tmp/cpu_extension.dll', 'CPU')       # absolute path is required
    net_det  = IENetwork(model=model_det+'.xml', weights=model_det+'.bin')
    input_name_det  = next(iter(net_det.inputs))                          # Input blob name "data"
    input_shape_det = net_det.inputs[input_name_det].shape                # [1,3,384,672]
    out_name_det    = next(iter(net_det.outputs))                         # Output blob name "detection_out"
    exec_net_det    = ie_det.load_network(network=net_det, device_name='CPU', num_requests=1)
    del net_det

    # Preparation for landmark detection
    ie_landmark = IECore()
    net_landmark = IENetwork(model=model_landmark+'.xml', weights=model_landmark+'.bin')
    input_name_landmark  = next(iter(net_landmark.inputs))                # Input blob name 
    input_shape_landmark = net_landmark.inputs[input_name_landmark].shape # [1,3,60,60]
    out_name_landmark    = next(iter(net_landmark.outputs))               # Output blob name "embd/dim_red/conv"
    out_shape_landmark   = net_landmark.outputs[out_name_landmark].shape  # 3x [1,1]
    exec_net_landmark    = ie_landmark.load_network(network=net_landmark, device_name='CPU', num_requests=1)
    del net_landmark

    # Preparation for headpose detection
    ie_hp = IECore()
    net_hp = IENetwork(model=model_hp+'.xml', weights=model_hp+'.bin')
    input_name_hp  = next(iter(net_hp.inputs))                            # Input blob name
    input_shape_hp = net_hp.inputs[input_name_hp].shape                   # [1,3,60,60]
    out_name_hp    = next(iter(net_hp.outputs))                           # Output blob name
    out_shape_hp   = net_hp.outputs[out_name_hp].shape                    # [1,70]
    exec_net_hp    = ie_hp.load_network(network=net_hp, device_name='CPU', num_requests=1)
    del net_hp

    # Preparation for gaze estimation
    ie_gaze = IECore()
    net_gaze = IENetwork(model=model_gaze+'.xml', weights=model_gaze+'.bin')
    if device == 'CPU':
      ie_gaze.add_extension(cpu_ext, 'CPU')    # absolute path is required
      #ie_gaze.add_extension('C:/tmp/cpu_extension.dll', 'CPU')       # absolute path is required
    #input_name_gaze  = next(iter(net_gaze.inputs))                       # Input blob name
    #input_shape_gaze = net_gaze.inputs[input_name_gaze].shape            # 2x [1,3,60,60]
    input_shape_gaze  = [1, 3, 60, 60]
    #out_name_gaze    = next(iter(net_gaze.outputs))                      # Output blob name
    #out_shape_gaze   = net_gaze.outputs[out_name_gaze].shape             # [1,3]
    exec_net_gaze     = ie_gaze.load_network(network=net_gaze, device_name='CPU')
    del net_gaze

    # Open USB webcams
    cam = cv2.VideoCapture(0)
    camx, camy = [(1920, 1080), (1280, 720), (800, 600), (480, 480)][1]
    cam.set(cv2.CAP_PROP_FRAME_WIDTH , camx)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

    laser_flag=True
    flip_flag =True
    spark_flag=True
    while True:
      ret,img = cam.read()                                                   # img won't be modified
      if flip_flag == True:
        img = cv2.flip(img, 1)                                               # flip image
      out_img = img.copy()                                                   # out_img will be drawn and modified to make an display image

      img1 = cv2.resize(img, (input_shape_det[_W], input_shape_det[_H]))
      img1 = img1.transpose((2, 0, 1))                                       # Change data layout from HWC to CHW
      img1 = img1.reshape(input_shape_det)
      res_det = exec_net_det.infer(inputs={input_name_det: img1})            # Detect faces

      gaze_lines = []
      for obj in res_det[out_name_det][0][0]:                                # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
        if obj[2] > 0.75:                                                    # Confidence > 75% 
          xmin = abs(int(obj[3] * img.shape[1]))
          ymin = abs(int(obj[4] * img.shape[0]))
          xmax = abs(int(obj[5] * img.shape[1]))
          ymax = abs(int(obj[6] * img.shape[0]))
          class_id = int(obj[1])
          face=img[ymin:ymax,xmin:xmax]                                         # Cut out the found object (face)
          if boundary_box_flag == True:
            cv2.rectangle(out_img, (xmin, ymin), (xmax, ymax), (255,255,0), 2)

          # Find facial landmarks (to find eyes)
          face1=cv2.resize(face, (input_shape_landmark[_W], input_shape_landmark[_H]))
          face1=face1.transpose((2,0,1))
          face1=face1.reshape(input_shape_landmark)
          res_landmark = exec_net_landmark.infer(inputs={input_name_landmark: face1}) # Run landmark detection
          lm=res_landmark[out_name_landmark][0][:8].reshape(4,2)    #  [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y] ]
          print(lm)

          # Estimate head orientation (yaw, pitch, role)
          res_hp = exec_net_hp.infer(inputs={input_name_hp: face1}) # Run head pose estimation
          yaw   = res_hp['angle_y_fc'][0][0]
          pitch = res_hp['angle_p_fc'][0][0]
          roll  = res_hp['angle_r_fc'][0][0]

          _X=0
          _Y=1
          eye_size   = [ abs(int((lm[0][_X]-lm[1][_X]) * face.shape[1])), abs(int((lm[3][_X]-lm[2][_X]) * face.shape[1])) ]    # eye size in the cropped face image
          print("eye size:", eye_size)
          eye_center = [ [ int(((lm[0][_X]+lm[1][_X])/2 * face.shape[1])), int(((lm[0][_Y]+lm[1][_Y])/2 * face.shape[0])) ], 
                         [ int(((lm[3][_X]+lm[2][_X])/2 * face.shape[1])), int(((lm[3][_Y]+lm[2][_Y])/2 * face.shape[0])) ] ]  # eye center coordinate in the cropped face image
          if eye_size[0]<4 or eye_size[1]<4:
            continue

          ratio = 0.7
          eyes = []
          for i in range(2):
            # Crop eye images
            x1 = int(eye_center[i][_X]-eye_size[i]*ratio)
            x2 = int(eye_center[i][_X]+eye_size[i]*ratio)
            y1 = int(eye_center[i][_Y]-eye_size[i]*ratio)
            y2 = int(eye_center[i][_Y]+eye_size[i]*ratio)
            print(x1, y2, x2, y2)
            eyes.append(cv2.resize(face[y1:y2, x1:x2].copy(), (input_shape_gaze[_W], input_shape_gaze[_H])))    # crop and resize

            # Draw eye boundary boxes
            if boundary_box_flag == True:
              x=eye_center[i][_X]
              y=eye_center[i][_Y]
              s=int(eye_size[i]*ratio)
              cv2.rectangle(out_img, (x-s+xmin, y-s+ymin), (x+s+xmin, y+s+ymin), (0,255,0), 2)

            # rotate eyes around Z axis to keep them level
            if roll != 0.:
              rotMat = cv2.getRotationMatrix2D((int(input_shape_gaze[_W]/2), int(input_shape_gaze[_H]/2)), roll, 1.0)
              eyes[i] = cv2.warpAffine(eyes[i], rotMat, (input_shape_gaze[_W], input_shape_gaze[_H]), flags=cv2.INTER_LINEAR)
              cv2.imshow(["eye-l", "eye-r"][i], eyes[i])

            #eyes[i] = cv2.resize(eyes[i], (input_shape_gaze[_W], input_shape_gaze[_H]))
            eyes[i] = eyes[i].transpose((2, 0, 1))                                       # Change data layout from HWC to CHW
            eyes[i] = eyes[i].reshape((1,3,60,60))

          #hp_angle = [ yaw, pitch, roll ]
          hp_angle = [ yaw, pitch, 0 ]                                                   # head pose angle in degree
          print("head pose angle:",hp_angle)
          res_gaze = exec_net_gaze.infer(inputs={'left_eye_image'  : eyes[0], 
                                                 'right_eye_image' : eyes[1],
                                                 'head_pose_angles': hp_angle})          # gaze estimation
          gaze_vec = res_gaze['gaze_vector'][0]                                          # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
          gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)                            # normalize vector
          print("gaze vector (orig/norm): ", gaze_vec, gaze_vec_norm)

          vcos = math.cos(math.radians(roll))
          vsin = math.sin(math.radians(roll))
          tmpx = gaze_vec_norm[0]*vcos + gaze_vec_norm[1]*vsin
          tmpy = gaze_vec_norm[0]*vsin + gaze_vec_norm[1]*vcos
          gaze_vec_norm = [tmpx, tmpy]

          # Store gaze line coordinations
          for i in range(2):
            coord1 = (eye_center[i][_X]+xmin, eye_center[i][_Y]+ymin)
            coord2 = (eye_center[i][_X]+xmin+int((gaze_vec_norm[0]+0.)*3000), eye_center[i][_Y]+ymin-int((gaze_vec_norm[1]+0.)*3000))
            gaze_lines.append([coord1, coord2, False])  # line(coord1, coord2); False=spark flag


      # Gaze lines intersection check
      if spark_flag == True:
        for g1 in range(len(gaze_lines)):
          for g2 in range(g1+1, len(gaze_lines)):
            if gaze_lines[g1][2]==True or gaze_lines[g2][2]==True:
              continue                 # Skip if either line has already marked as crossed
            x1 = gaze_lines[g1][0]
            y1 = gaze_lines[g1][1]
            x2 = gaze_lines[g2][0]
            y2 = gaze_lines[g2][1]
            if intersection_check(x1, y1, x2, y2) == True:
              l1 = line(x1, y1)
              l2 = line(x2, y2) 
              x, y = intersection( l1, l2 )           # calculate crossing coordinate
              gaze_lines[g1][1] = [int(x), int(y)]
              gaze_lines[g1][2] = True
              gaze_lines[g2][1] = [int(x), int(y)]
              gaze_lines[g2][2] = True

      # Drawing gaze lines and sparks
      for gaze_line in gaze_lines:
        draw_gaze_line(out_img, (gaze_line[0][0], gaze_line[0][1]), (gaze_line[1][0], gaze_line[1][1]), laser_flag)
        if gaze_line[2]==True:
          draw_spark(out_img, (gaze_line[1][0], gaze_line[1][1]))

      cv2.imshow("gaze", out_img)

      key = cv2.waitKey(1)
      if key == 27: break
      if key == ord(u'l'): laser_flag        = True if laser_flag       == False else False    # toggles laser_flag
      if key == ord(u'f'): flip_flag         = True if flip_flag        == False else False    # image flip flag
      if key == ord(u'b'): boundary_box_flag = True if boundary_box_flag== False else False    # boundary box flag
      if key == ord(u's'): spark_flag        = True if spark_flag       == False else False    # spark flag

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)

