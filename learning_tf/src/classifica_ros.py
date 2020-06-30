#!/usr/bin/env python
import rospy

import math
import tf2_ros
import geometry_msgs.msg
from astra_body_tracker.msg import BodyListStamped

frame_names = ["head_", "neck_", "torso_", "left_shoulder_", "right_shoulder_", "left_hand_", "right_hand_",
               "left_elbow_", "right_elbow_", "left_hip_", "right_hip_", "left_knee_", "right_knee_", "left_foot_",
               "right_foot_"]

body_transformations = []
camera_transformations = []


def body_list_callback(data):
    msg = BodyListStamped()
    msg = data

    for id in msg.ids:
        try:
            trans = tfBuffer.lookup_transform('', '', msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('classifica_ros')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber("body_list", BodyListStamped, body_list_callback)

    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        rate.sleep()
