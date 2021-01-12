# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This is ROS action server to record tactile data with the Jaco arm on the Doro robot sensorized with 3 OptoForce sensors.

# #### You can freely modify this code for your own purpose. However, please cite this work when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Philip Maus (philiprmaus@gmail.com)
# 
# https://github.com/philipmaus/Tactile_Object_Classification_with_sensorized_Jaco_arm_on_Doro_robot


#! /usr/bin/env python
import rospy
import time
import actionlib
import numpy as np
import csv
import os
import math
from thesis.msg import RecDataFeedback, RecDataResult, RecDataAction
from geometry_msgs.msg import WrenchStamped
from kinova_msgs.msg import FingerPosition
from kinova_msgs.msg import SetFingersPositionAction, SetFingersPositionGoal, SetFingersPositionResult, SetFingersPositionFeedback
from kinova_msgs.msg import ArmJointAnglesAction, ArmJointAnglesGoal, ArmJointAnglesResult, ArmJointAnglesFeedback

class RecordDataClass(object):
    
  # create messages that are used to publish feedback/result
  _feedback = RecDataFeedback()
  _result   = RecDataResult()

  def __init__(self):
    # create the action server
    self._as = actionlib.SimpleActionServer("/record_data_as", RecDataAction, self.goal_callback, False)
    self._as.start()
    self.ctrl_c = False
    self.rate = rospy.Rate(1)

    # define all topic subscriber and publisher
    #optoforce
    self._subOpto1 = rospy.Subscriber('/optoforce_3', WrenchStamped, self.callback_opto_1)
    self._subOpto2 = rospy.Subscriber('/optoforce_1', WrenchStamped, self.callback_opto_2)
    self._subOpto3 = rospy.Subscriber('/optoforce_2', WrenchStamped, self.callback_opto_3)
    #finger position
    self._subKG3_fing_pos = rospy.Subscriber('/j2s6s300_driver/out/finger_position', FingerPosition, self.callback_kg3_fing_pos)
    #move gripper fingers
    action_address = '/j2s6s300_driver/fingers_action/finger_positions'
    self._clientMoveFinger = actionlib.SimpleActionClient(action_address, SetFingersPositionAction)
    self._clientMoveFinger.wait_for_server()
    #move whole arm
    action_address_arm = '/j2s6s300_driver/joints_action/joint_angles'
    self._clientMoveArm = actionlib.SimpleActionClient(action_address_arm, ArmJointAnglesAction)
    self._clientMoveArm.wait_for_server()

    # define variables
    self.OptoForce_1 = []
    self.OptoForce_2 = []
    self.OptoForce_3 = []
    self._setFingerPos = SetFingersPositionGoal() 
    self._setArmJointPos = ArmJointAnglesGoal()
    self.Gripper_FingerPosition = FingerPosition()
    self.finger_positions = [0,0,0]
    self.data_buf = []
    self.data = []					# storage for measurement data
    self.measNumber = 0				# number of measurements to be taken
    self.class_label = 20			# total number of classes/objects
    self.start_stamp = 0			# stamp of measurement starting time
    self.fingpos_open = 20			# open gripper position
    self.fingpos_closed = 52		# closed gripper position
    self.force_limit_12 = 2			# if euclidean norm of force sensor > limit, dont close finger any further
    self.force_limit_3 = 1 

            
  def callback_opto_1(self,msg):
    self.OptoForce_1 = msg.wrench.force
    rospy.loginfo("OptoForce Sensor 1 Callback called: "+str(self.OptoForce_1))

  def callback_opto_2(self,msg):
    self.OptoForce_2 = msg.wrench.force
    rospy.loginfo("OptoForce Sensor 2 Callback called: "+str(self.OptoForce_2))

  def callback_opto_3(self,msg):
    self.OptoForce_3 = msg.wrench.force
    rospy.loginfo("OptoForce Sensor 3 Callback called: "+str(self.OptoForce_3))

  def callback_kg3_fing_pos(self,msg):
    self.Gripper_FingerPosition = msg
    #rospy.loginfo("KG-3 Gripper Finger Position Callback called: "+str(self.Gripper_FingerPosition))

  def set_finger_position(self,finger_positions):
    self._setFingerPos.fingers.finger1 = float(finger_positions[0])
    self._setFingerPos.fingers.finger2 = float(finger_positions[1])
    self._setFingerPos.fingers.finger3 = float(finger_positions[2])
    self._clientMoveFinger.send_goal(self._setFingerPos)

  # function that resets gripper in initial open position
  def reset_gripper(self):
    rospy.loginfo("Resetting gripper...")
    self.finger_positions[0] = self.fingpos_open
    self.finger_positions[1] = self.fingpos_open
    self.finger_positions[2] = self.fingpos_open
    i=0
    while not i == 3:
      self.set_finger_position(self.finger_positions)   # call setFingPos action server
      time.sleep(1)
      i += 1

  # function that closes the gripper stepwise if force measured under certain limit value
  def move_gripper(self,i):
    self.fingpos_step = float(float(self.fingpos_closed-self.fingpos_open)/self.measNumber)
    rospy.loginfo("eucl. force finger 1: " + str(math.sqrt(self.OptoForce_1.x**2+self.OptoForce_1.y**2+self.OptoForce_1.z**2)))
    rospy.loginfo("eucl. force finger 2: " + str(math.sqrt(self.OptoForce_2.x**2+self.OptoForce_2.y**2+self.OptoForce_2.z**2)))
    rospy.loginfo("eucl. force finger 3: " + str(math.sqrt(self.OptoForce_3.x**2+self.OptoForce_3.y**2+self.OptoForce_3.z**2)))

    if math.sqrt(self.OptoForce_1.x**2+self.OptoForce_1.y**2+self.OptoForce_1.z**2) < self.force_limit_12 and math.sqrt(self.OptoForce_2.x**2+self.OptoForce_2.y**2+self.OptoForce_2.z**2) < self.force_limit_12:
      self.finger_positions[0] += self.fingpos_step
      self.finger_positions[1] += self.fingpos_step
    else:
      self.finger_positions[0] -= self.fingpos_step
      self.finger_positions[1] -= self.fingpos_step
      rospy.loginfo("eucl 1 and 2 too high")
    if math.sqrt(self.OptoForce_3.x**2+self.OptoForce_3.y**2+self.OptoForce_3.z**2) < self.force_limit_3:
      self.finger_positions[2] += self.fingpos_step
    else:
      self.finger_positions[2] -= self.fingpos_step
      rospy.loginfo("eucl 3 too high")
    self.set_finger_position(self.finger_positions)   # call setFingPos action server
    #rospy.loginfo("finger position: " + str(self.finger_positions))
    

  def get_measurement_data(self,i):
    self.data_buf = [i+1,\
    				 self.OptoForce_1.x, self.OptoForce_1.y, self.OptoForce_1.z,\
    				 self.OptoForce_2.x, self.OptoForce_2.y, self.OptoForce_2.z,\
    				 self.OptoForce_3.x, self.OptoForce_3.y, self.OptoForce_3.z,\
    				 self.Gripper_FingerPosition.finger1, self.Gripper_FingerPosition.finger2, self.Gripper_FingerPosition.finger3]
    self.data = np.append(self.data, self.data_buf)

  def set_init_position(self):
    rospy.loginfo('Initialize position')
    #initialize arm
    self._setArmJointPos.angles.joint1 = 250
    self._setArmJointPos.angles.joint2 = 220
    self._setArmJointPos.angles.joint3 = 70
    self._setArmJointPos.angles.joint4 = 150
    self._setArmJointPos.angles.joint5 = 120
    self._setArmJointPos.angles.joint6 = 50
    self._setArmJointPos.angles.joint7 = 0
    self._clientMoveArm.send_goal(self._setArmJointPos)

  def send_feedback(self,i):
    self._feedback.feedback = i
    self._as.publish_feedback(self._feedback)

  def send_result(self):
    self._result.header.seq = 0
    self._result.header.stamp = rospy.Time.now()
    self._result.header.frame_id = ''
    self._result.data = self.data
    rospy.loginfo('Data recording complete')
    self._as.set_succeeded(self._result)

  def save_data(self):
    if os.getcwd() != '/home/philip/catkin_ws/src/thesis/data':
      os.chdir('../catkin_ws/src/thesis/data/')   # set working directory
    rospy.loginfo('Working directory: ' + os.getcwd())
    self.data = np.reshape(self.data,(self.measNumber, self.data.shape[0]/self.measNumber))
    filepath =  str(self.class_label) + '_' + str(self._result.header.stamp) + '.csv'
    with open(filepath, 'w') as f:
      thewriter = csv.writer(f)
      thewriter.writerow(['header',\
      					  'seq', self._result.header.seq,\
      					  'stamp start', self.start_stamp,\
      					  'stamp end', self._result.header.stamp,\
      					  'nMeas', self.measNumber,\
      					  'class label', self.class_label])
      thewriter.writerow(['timestep',\
                          'opto1_x','opto1_y','opto1_z',\
                          'opto2_x','opto2_y','opto2_z',\
                          'opto3_x','opto3_y','opto3_z',\
                          'finger1_pos','finger2_pos','finger3_pos'])
      for i in xrange(0, self.measNumber):
        thewriter.writerow(self.data[i,:])
      f.close()
    rospy.loginfo("Data saved as " + filepath)
    

  def goal_callback(self, goal):
    # define parameters
    self.data = []    # empty data array
    self.measNumber = goal.nMeas
    self.class_label = goal.label
    self.start_stamp = rospy.Time.now()
    success = True
    gripperSeconds = 0.2
    r = rospy.Rate(5)

    # initialize arm position
    self.set_init_position()
    time.sleep(10)

    # reset gripper
    self.reset_gripper()

    # data acquisition loop
    rospy.loginfo("Acquire data...")
    for i in xrange(0, self.measNumber):
    
      # check that preempt has not been requested by the action client
      if self._as.is_preempt_requested():
        rospy.loginfo('The goal has been cancelled/preempted')
        self._as.set_preempted()
        success = False
        break
    
      # close gripper one step
      self.move_gripper(i)
      time.sleep(gripperSeconds)  # wait for gripper motion to be executed

      # get measurement data from OptoForce sensors and gripper actuators
      self.get_measurement_data(i)
      
      # build and publish the feedback message: current measurement number
      self.send_feedback(i)
      r.sleep()

    # send result
    if success:
      self.send_result()
        
    # reset gripper to initial position
    self.reset_gripper()

    #save data
    if success:
      self.save_data()

      
if __name__ == '__main__':
  rospy.init_node('record_data_as_node')
  RecordDataClass()
  rospy.spin()