# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This is ROS action client to record tactile data with the Jaco arm on the Doro robot sensorized with 3 OptoForce sensors.

# #### You can freely modify this code for your own purpose. However, please cite this work when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Philip Maus (philiprmaus@gmail.com)
# 
# https://github.com/philipmaus/Tactile_Object_Classification_with_sensorized_Jaco_arm_on_Doro_robot


#! /usr/bin/env python
import rospy
import time
import actionlib
import numpy as np
import sys
from thesis.msg import RecDataAction, RecDataGoal, RecDataResult, RecDataFeedback


PENDING = 0
ACTIVE = 1
DONE = 2
WARN = 3
ERROR = 4

nMeas = 50		# define number of measurements
class_label = 20


def feedback_callback(feedback):
    print('[Feedback] received', feedback)

if __name__ == '__main__':
	# initializes the action client node
	rospy.init_node('record_data_ac_node')

	# create the connection to the action server
	action_server_name = "record_data_as"
	client = actionlib.SimpleActionClient(action_server_name, RecDataAction)

	# wait until the action server is up and running
	rospy.loginfo('Waiting for action Server '+action_server_name)
	client.wait_for_server()
	rospy.loginfo('Action Server Found...'+action_server_name)

	# create a goal to send to the action server
	goal = RecDataGoal()
	args = rospy.myargv(argv=sys.argv)
	if len(args) != 2:
		rospy.loginfo("ERROR: NO CLASS LABEL PROVIDED")
		rospy.loginfo("Usage: roslaunch thesis call_record_data.launch class_number:=")
		sys.exit(1)
	class_label = int(args[1])
	if class_label < 0 or class_label > 19:
	        print("Class label has in range 0 to 15")
	        sys.exit (1)
	goal.label = class_label
	goal.nMeas = nMeas
	client.send_goal(goal, feedback_cb=feedback_callback)
	rospy.loginfo('Goal command sent...')

	# wait for action server to complete measurement
	rate = rospy.Rate(10)
	state_result = client.get_state()
	while state_result < DONE:
	    rate.sleep()
	    state_result = client.get_state()

	# get action topic result
	result = client.get_result()
	rospy.loginfo("Final action result received: [State]: "+str(state_result))
	if state_result == ERROR:
	    rospy.logerr("Something went wrong in the Server Side")
	if state_result == WARN:
	    rospy.logwarn("There is a warning in the Server Side")