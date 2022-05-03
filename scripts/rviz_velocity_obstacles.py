#!/usr/bin/env python3
from re import A
from turtle import position
import rospy 
import math
from geometry_msgs.msg import Twist #Velocity 
from nav_msgs.msg import Odometry ##Position 
from tf.transformations import euler_from_quaternion
import numpy as np 
from scipy.optimize import Bounds, minimize
import time 

class mobile_robot:
    def __init__(self):

        # Adhoc paramters
        self.t_step = 0.05
        self.ra_rb = 0.3
        self.v_max = 3
        self.v_min = 0
        #Setting of the nodes 
        rospy.init_node("mobile_robot_move")
        rospy.loginfo("Press Ctrl + C to terminate")
        self.velocity_pub = rospy.Publisher("/robot_0/cmd_vel", Twist,queue_size=10)
        self.rate = rospy.Rate(100)
        self.logging_counter = 0
        self.robot_z = 0

        # States
        self.curr_state = np.asarray([2.0,3.0])  # Start position right,up
        self.goal_state = np.asarray([14.4, 2.779])
        self.rbt_vel = [0,0] 
        
        # Metrics
        self.prev_state = self.curr_state.copy()
        self.path_length = 0
        self.min_distance = self.dis(self.curr_state,self.goal_state )
        self.start_time = time.time() #This is the start time
        self.distance_to_obstacle  = dict()
        self.min_distance_tracker = {"1":np.inf}

        rospy.Subscriber("/robot_0/odom", Odometry, self.robot_position) #subscribe to robot position 
        self.obstacle_vel_topics = []
        self.obstacle_pos_topics = []

        for i in range(1,14):
            vel_topic = "/robot_"+str(i)+"/cmd_vel"
            pos_topic = "/robot_"+str(i)+"/odom"
            self.obstacle_vel_topics.append(vel_topic)
            self.obstacle_pos_topics.append(pos_topic)

        self.obstacle_vel_data = {}
        self.obstacle_pos_data = {}


        for obs_vel_topic,obs_pos_topic in zip(self.obstacle_vel_topics,self.obstacle_pos_topics):
            rospy.Subscriber(obs_vel_topic, Twist, callback = self.obstacle_velocity_callback,callback_args=obs_vel_topic) #subscribe to robot position 
            rospy.Subscriber(obs_pos_topic, Odometry, callback = self.obstacle_position_callback,callback_args=obs_pos_topic) #subscribe to robot position 

            key = obs_vel_topic.split("_")[1].split("/")[0]
            self.obstacle_vel_data[key] = [0,0]  # Initializing the dict with X Y velocities 
            self.obstacle_pos_data[key] = [0,0]

        try:
            self.run()
        except Exception as err:
            print(err)


    def dis(self,node1,node2):
        return np.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)

    def obstacle_velocity_callback(self,data,topic):
        key = topic.split("_")[1].split("/")[0]
        x_vel = data.linear.x
        y_vel = data.linear.y
        self.obstacle_vel_data[key] = [x_vel,y_vel]

    def obstacle_position_callback(self,data,topic):
        key = topic.split("_")[1].split("/")[0]
        x_pos = data.pose.pose.position.x
        y_pos = data.pose.pose.position.y
        self.obstacle_pos_data[key] = [x_pos,y_pos]

        for key in self.obstacle_pos_data:
            self.distance_to_obstacle[key] = self.dis(self.curr_state, self.obstacle_pos_data[key])
        # Finding the minimum distance to the obstacle at each update 
        min_key = min(zip(self.distance_to_obstacle.values(), self.distance_to_obstacle.keys()))[1]
        if self.distance_to_obstacle[min_key] < list(self.min_distance_tracker.values())[0]:
            # Here storing the minum value in a dictionary that can be acquired later
            self.min_distance_tracker[min_key] = self.distance_to_obstacle[min_key]

    def cmpt_des_vel(self):
        disp_vec = self.goal_state - self.curr_state
        norm = np.linalg.norm(disp_vec)
        # print(norm)
        if norm < self.ra_rb/ 5:
            desired_vel = np.zeros(2)
        
        else:
            disp_vec = disp_vec / norm
            desired_vel = self.v_max * disp_vec

        # print(desired_vel)
        return desired_vel
    
    def cmpt_vel(self, v_desired):

        pA = self.curr_state
        vA = self.rbt_vel
        # Compute the constraints
        # for each velocity obstacles
        number_of_obstacles = len(self.obstacle_pos_data)
        Amat = np.empty((number_of_obstacles * 2, 2))
        bvec = np.empty((number_of_obstacles * 2))


        for key in  self.obstacle_pos_data:
            pB = self.obstacle_pos_data[key]
            vB = self.obstacle_vel_data[key]

            dispBA = pA - pB
            distBA = np.linalg.norm(dispBA)
            thetaBA = np.arctan2(dispBA[1], dispBA[0])
            if 2.2 * self.ra_rb > distBA:
                distBA = 2.2*self.ra_rb
            phi_obst = np.arcsin(2.2*self.ra_rb/distBA)
            phi_left = thetaBA + phi_obst
            phi_right = thetaBA - phi_obst


            # VO
            idx = int(key)-1
            translation = np.asarray(vB)
            Atemp, btemp = self.create_own_cts(translation, phi_left, "left")
            Amat[idx*2, :] = Atemp
            bvec[idx*2] = btemp
            Atemp, btemp = self.create_own_cts(translation, phi_right, "right")
            Amat[idx*2 + 1, :] = Atemp
            bvec[idx*2 + 1] = btemp

        # Create search-space
        th = np.linspace(0, 2*np.pi, 20)
        vel = np.linspace(self.v_min, self.v_max, 5)

        vv, thth = np.meshgrid(vel, th)

        vx_sample = (vv * np.cos(thth)).flatten()
        vy_sample = (vv * np.sin(thth)).flatten()

        v_sample = np.stack((vx_sample, vy_sample))

        v_satisfying_constraints = self.chk_contraints(v_sample, Amat, bvec)

        # Objective function
        size = np.shape(v_satisfying_constraints)[1]
        diffs = v_satisfying_constraints - \
            ((v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size))
        norm = np.linalg.norm(diffs, axis=0)
        # print(norm)
        min_index = np.where(norm == np.amin(norm))[0][0]
        cmd_vel = (v_satisfying_constraints[:, min_index])

        # print(cmd_vel)

        return cmd_vel

    def chk_contraints(self,v_sample, Amat, bvec):
        length = np.shape(bvec)[0]

        for i in range(int(length/2)):
            v_sample = self.check_inside(v_sample, Amat[2*i:2*i+2,:], bvec[2*i:2*i+2])
        return v_sample

    def check_inside(self, v, Amat, bvec):
        v_out = []
        for i in range(np.shape(v)[1]):
            if not ((Amat @ v[:, i] < bvec).all()):
                v_out.append(v[:, i])
        return np.array(v_out).T

    def create_own_cts(self, translation, angle, side):
        # create line
        origin = np.array([0, 0, 1])
        point = np.array([np.cos(angle), np.sin(angle)])
        line = np.cross(origin, point)
        line = self.line_trns(line, translation)

        if side == "left":
            line *= -1

        A = line[:2]
        b = -line[2]

        return A, b

    def line_trns(self, line, translation):
        matrix = np.eye(3)
        matrix[2, :2] = -translation[:2]
        return matrix @ line

    def repl_state(self, x, v):
        new_state_pos = x + v * self.t_step
        new_state_vel = v

        return new_state_pos, new_state_vel

    def robot_position(self,data):
        self.curr_state = np.asarray([data.pose.pose.position.x,data.pose.pose.position.y])
        (q , w , z) = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
        self.robot_z = z
        self.path_length += self.dis(self.curr_state ,self.prev_state)
        self.prev_state = self.curr_state.copy()

    def run(self):
        self.flag = True
        while not rospy.is_shutdown():
            self.vel_data =Twist()
            
        
            v_desired = self.cmpt_des_vel()
            self.control_vel = self.cmpt_vel(v_desired)# RECTIFY
            self.curr_state, self.rbt_vel = self.repl_state(self.curr_state, self.control_vel)         
            

            # print(velocity)

            heading = math.atan2(self.rbt_vel[1], self.rbt_vel[0])            

            if self.robot_z - heading >= 0 :
                dir = -1 * heading
                #print('negative')
            else :
                dir = 1 * heading

            if abs(self.robot_z - heading) >= 0.1:
                self.vel_data.linear.x = np.sqrt(self.rbt_vel[0]**2 + self.rbt_vel[1]**2)
                self.vel_data.angular.z = 0.5 * dir
                if dir >=0 and self.rbt_vel[1] < 0:
                    self.vel_data.angular.z = -0.5 
            else:
                self.vel_data.linear.x = np.sqrt(self.rbt_vel[0]**2 + self.rbt_vel[1]**2)
                self.vel_data.angular.z = 0.1 * dir
            

            if self.goal_state[0]-0.5 <=  self.curr_state[0] <= self.goal_state[0]+0.5 and \
            self.goal_state[1]-0.5 <=  self.curr_state[1] <= self.goal_state[1]+0.5 :
               
                self.vel_data.linear.x = 0
                self.vel_data.angular.z = 0

                if self.flag:
                    print("Path length travelled by robot")
                    print(self.path_length)
                    print("Distance ratio")
                    print(self.path_length/self.min_distance)
                    print("Total time taken by robot is:")
                    print(time.time()-self.start_time)
                    print("Closest obstacle is")
                    min_key = min(zip(self.distance_to_obstacle.values(), self.distance_to_obstacle.keys()))[1]
                    print(list(self.min_distance_tracker.keys())[0])
                    print("Closest distance was")
                    print(list(self.min_distance_tracker.values())[0])
                    self.flag = False

            self.velocity_pub.publish(self.vel_data)
            self.rate.sleep()
        self.stop()

if __name__ == "__main__":
    mobile_robot()
