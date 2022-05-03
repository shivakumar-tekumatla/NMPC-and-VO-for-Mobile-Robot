#!/usr/bin/env python
from turtle import position
import rospy 
import math
from geometry_msgs.msg import Twist #Velocity 
from nav_msgs.msg import Odometry ##Position 
from tf.transformations import euler_from_quaternion
import numpy as np 
from scipy.optimize import Bounds, minimize 
import time

# class Controller:
#     def __init__(self):
#         return None 

class mobile_robot:
    def __init__(self):
        # """Initially publishing """
        # now = rospy.Time.now()
        # self.robot_start_pos_topic = "/robot_0/odom"
        # self.robot_start = Odometry()
        # self.robot_start.header.frame_id = self.robot_start_pos_topic
        # self.robot_start.header.stamp = now
        # self.robot_start.child_frame_id = " "
        # self.robot_start.pose.pose.position.x = 10
        # self.robot_start.pose.pose.position.y = 10
        """Setting up NMPC parameters """
        self.SIMULATION_TIME = 2
        self.TIMESTEP = 0.1
        self.NUMBER_OF_TIMESTEPS = int(self.SIMULATION_TIME/self.TIMESTEP)
        self.ROBOT_RADIUS = 0.8
        self.VEL_MAX = 2.0
        self.VEL_MIN = 0.1
        # collision cost parameters
        self.Collision_Cost = 1.0
        self.Collision_Radius = 9

        # nmpc parameters
        self.HORIZON = int(4)
        self.NMPC_STEP = 0.3
        self.u_bound = [(1/np.sqrt(2)) * self.VEL_MAX] * self.HORIZON * 2
        self.l_bound = [-(1/np.sqrt(2)) * self.VEL_MAX] * self.HORIZON * 2

        #Setting of the nodes 
        rospy.init_node("mobile_robot_move")
        rospy.loginfo("Press Ctrl + C to terminate")
        self.velocity_pub = rospy.Publisher("/robot_0/cmd_vel", Twist,queue_size=10)
        self.rate = rospy.Rate(100)
        self.logging_counter = 0
        self.robot_z = 0
        self.robot_state = np.asarray([2.0,3.0])  # Start position right,up
        self.prev_state = self.robot_state.copy()
        self.path_length = 0 # Initially this path is zero. This variable tracks the total distance travelled by the robot 
        self.robot_goal_position = np.asarray([19, 3.079]) # Goal #Robot goal poisition 
        self.min_distance = self.dis(self.robot_state,self.robot_goal_position )
        self.start_time = time.time() #This is the start time
        self.distance_to_obstacle  = dict()
        self.min_distance_tracker = {"1":np.inf}
        """Way points"""
        # self.way_points = [[9.2,3.5],[14,1.5],[16.5,3.7],[20,4.5],self.robot_goal_position]
        # self.way_inc = 0
        # self.robot_goal_position = self.way_points[0]
        """Way Points END"""
        rospy.Subscriber("/robot_0/odom", Odometry, self.robot_position) #subscribe to robot position 
        self.obstacle_vel_topics = []
        self.obstacle_pos_topics = []
        for i in range(1,14):
            vel_topic = "/robot_"+str(i)+"/cmd_vel"
            pos_topic = "/robot_"+str(i)+"/odom"
            self.obstacle_vel_topics.append(vel_topic)
            self.obstacle_pos_topics.append(pos_topic)
        # self.obstacle_vel_topics = ["/robot_1/cmd_vel","/robot_2/cmd_vel",
        #                             "/robot_3/cmd_vel","/robot_4/cmd_vel",
        #                             "/robot_5/cmd_vel","/robot_6/cmd_vel",
        #                             "/robot_7/cmd_vel","/robot_8/cmd_vel",
        #                             "/robot_9/cmd_vel","/robot_10/cmd_vel",
        #                             "/robot_11/cmd_vel","/robot_12/cmd_vel",
        #                             "/robot_13/cmd_vel"]#,"/robot_14/cmd_vel",
        #                             # "/robot_15/cmd_vel","/robot_16/cmd_vel",
        #                             # "/robot_17/cmd_vel","/robot_18/cmd_vel"]
        # self.obstacle_pos_topics = ["/robot_1/odom","/robot_2/odom","/robot_3/odom",
        #                             "/robot_4/odom","/robot_5/odom","/robot_6/odom",
        #                             "/robot_7/odom","/robot_8/odom","/robot_9/odom",
        #                             "/robot_10/odom","/robot_11/odom","/robot_12/odom",
        #                             "/robot_13/odom"]#,"/robot_14/odom",
        #                             # "/robot_15/odom","/robot_16/odom",
        #                             # "/robot_17/odom","/robot_18/odom"]
        self.obstacle_vel_data = {}
        self.obstacle_pos_data = {}
        # """Static Obstacles """
        # static_obstacles = []
        # y_vals = np.linspace(0,6,30)
        # for i in y_vals:
        #     static_obstacles.append([6,i])
        # key_max = 14
        # for i in range(0,len(static_obstacles)):
        #     self.obstacle_pos_data[str(key_max)] = static_obstacles[i]
        #     self.obstacle_vel_data[str(key_max)] = [0,0]
        #     key_max+=1
        # """static end"""
        for obs_vel_topic,obs_pos_topic in zip(self.obstacle_vel_topics,self.obstacle_pos_topics):
            rospy.Subscriber(obs_vel_topic, Twist, callback = self.obstacle_velocity_callback,callback_args=obs_vel_topic) #subscribe to robot position 
            rospy.Subscriber(obs_pos_topic, Odometry, callback = self.obstacle_position_callback,callback_args=obs_pos_topic) #subscribe to robot position 

            key = obs_vel_topic.split("_")[1].split("/")[0]
            self.obstacle_vel_data[key] = [0,0]  # Initializing the dict with X Y velocities 
            self.obstacle_pos_data[key] = [0,0]
        # x_max = 25
        # y_max = 6 

        # static_obstacle_poisition = []
        # x_static = np.linspace(0,x_max,0.5)
        # y_static = np.linspace(0,y_max,0.5)
        # for i in x_static:
        #     static_obstacle_poisition.append([i,0])
        #     static_obstacle_poisition.append([i,y_max])

        # for i in y_static:
        #     static_obstacle_poisition.append([0,i])
        #     static_obstacle_poisition.append([x_max,i])

        # key_max =14
        # for i in range(0,len(static_obstacle_poisition)):
        #     self.obstacle_pos_data[str(key_max)] = static_obstacle_poisition[i]
        #     self.obstacle_vel_data[str(key_max)] = [0,0]
        #     key_max+=1

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
            self.distance_to_obstacle[key] = self.dis(self.robot_state, self.obstacle_pos_data[key])
        # Finding the minimum distance to the obstacle at each update 
        min_key = min(zip(self.distance_to_obstacle.values(), self.distance_to_obstacle.keys()))[1]
        if self.distance_to_obstacle[min_key] < list(self.min_distance_tracker.values())[0]:
            # Here storing the minum value in a dictionary that can be acquired later
            self.min_distance_tracker[min_key] = self.distance_to_obstacle[min_key]

    def future_obstacle_positions(self):

        self.obstacle_predictions = {}
        dyn_keys = [str(i+1) for i in range(13)]

        for key in  self.obstacle_pos_data:
            position = self.obstacle_pos_data[key]
            velocity =  self.obstacle_vel_data[key]
            u=[]
            for _ in range(self.HORIZON):
                u.append(velocity[0])
                u.append(velocity[1])
            u = np.asarray(u)
            N = int(len(u) / 2)
            u_pos = [] 
            for _ in range(N):
                u_pos.append(position[0])
                u_pos.append(position[1])
            u_pos = np.asarray(u_pos)
            kron = []
            for i in range(self.HORIZON):
                kron.append(velocity[0]*(i+1))
                kron.append(velocity[1]*(i+1))
            kron = np.asarray(kron)
            new_state = u_pos + kron*self.NMPC_STEP
            self.obstacle_predictions[key] = new_state 
    def reference_trajectory(self):
        dir_vec = self.robot_goal_position- self.robot_state
        norm = np.linalg.norm(dir_vec)
        if norm < 0.1:
            new_goal = self.robot_state
        else:
            dir_vec = dir_vec / norm
            new_goal = self.robot_state + dir_vec * self.VEL_MAX * self.NMPC_STEP * self.HORIZON
        return np.linspace(self.robot_state, new_goal, self.HORIZON).reshape((2*self.HORIZON))
    def overall_cost(self,u): 
        return self.total_cost(u)
    def velocity_required(self):
        u0 = np.random.rand(2*self.HORIZON)
        bounds = Bounds(self.l_bound, self.u_bound)
        res = minimize(self.overall_cost, u0, method='SLSQP', bounds=bounds)
        velocity = res.x[:2]
        return velocity, res.x

    def total_cost(self,u):
        N = int(len(u) / 2)
        u_pos = [] 
        position = self.robot_state
        for _ in range(N):
            u_pos.append(position[0])
            u_pos.append(position[1])
        u_pos = np.asarray(u_pos)
        kron = []
        for i in range(self.HORIZON):
            kron.append(u[0]*(i+1))
            kron.append(u[1]*(i+1))
        kron = np.asarray(kron)
        x_robot = u_pos + kron*self.NMPC_STEP
        c1 = self.tracking_cost(x_robot, self.xref)
        c2 = self.total_collision_cost(x_robot, self.obstacle_predictions_array)
        total = c1 + c2
        return total
    def tracking_cost(self,x,xref):
        return np.linalg.norm(x-xref)

    def total_collision_cost(self,robot, obstacles):
        total_cost = 0
        for i in range(self.HORIZON):
            for j in range(len(obstacles)):
                obstacle = obstacles[j]
                rob = robot[2 * i: 2 * i + 2]
                obs = obstacle[2 * i: 2 * i + 2]
                total_cost += self.collision_cost(rob, obs)
        return total_cost
    def collision_cost(self,x0, x1):
        """
        Cost of collision between two robot_state
        """
        d = np.linalg.norm(x0 - x1)
        cost = self.Collision_Cost / (1 + np.exp(self.Collision_Radius * (d - 2*self.ROBOT_RADIUS)))
        return cost
    def robot_position(self,data):
        self.robot_state = np.asarray([data.pose.pose.position.x,data.pose.pose.position.y])
        (q , w , z) = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
        self.robot_z = z
        self.path_length += self.dis(self.robot_state ,self.prev_state)
        self.prev_state = self.robot_state.copy()
        # print(self.robot_state)
    def run(self):
        self.flag = True
        while not rospy.is_shutdown():
            self.vel_data =Twist()
            
            # print("Velocity")
            # print(self.obstacle_vel_data)
            # print("Position")
            # print(self.obstacle_pos_data)
            self.future_obstacle_positions()
            self.xref = self.reference_trajectory()
            self.obstacle_predictions_array = np.asarray([self.obstacle_predictions[i] for i in self.obstacle_predictions])
            # print(self.obstacle_predictions_array)
            velocity, res = self.velocity_required()

            heading = math.atan2(velocity[1], velocity[0])            

            if self.robot_z - heading >= 0 :
                dir = -1 * heading
                #print('negative')
            else :
                dir = 1 * heading

            if abs(self.robot_z - heading) >= 0.1:
                self.vel_data.linear.x = np.sqrt(velocity[0]**2 + velocity[1]**2)
                self.vel_data.angular.z = 0.5 * dir
                if dir >=0 and velocity[1] < 0:
                    self.vel_data.angular.z = -0.5 
            else:
                self.vel_data.linear.x = np.sqrt(velocity[0]**2 + velocity[1]**2)
                self.vel_data.angular.z = 0.1 * dir
            

            if self.robot_goal_position[0]-0.5 <=  self.robot_state[0] <= self.robot_goal_position[0]+0.5 and \
            self.robot_goal_position[1]-0.5 <=  self.robot_state[1] <= self.robot_goal_position[1]+0.5 :
                self.vel_data.linear.x = 0
                self.vel_data.angular.z = 0
                # if self.way_inc == len(self.way_points)-1:
                #     self.vel_data.linear.x = 0
                #     self.vel_data.angular.z = 0
                # else:
                #     self.way_inc +=1 
                #     self.robot_goal_position = self.way_points[self.way_inc]
                
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
                    # break 

            self.velocity_pub.publish(self.vel_data)
            self.rate.sleep()
        self.stop()

if __name__ == "__main__":
    mobile_robot()

