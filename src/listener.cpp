#include <ros/ros.h>
#include <rosbag/bag.h>

#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <theora_image_transport/Packet.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/State.h>
#include <neptune/Custom.h>

#include <ctime>


/** A Listener Class **/
class Listener
{
    public:

    /** Initial **/
    Listener(char *filename, double loop_rate)
    {       
        _bag.open(filename, rosbag::bagmode::Write);

        _loop_rate = loop_rate;

        _color_sub  = _n.subscribe("/camera/color/image_raw", 10, &Listener::updateColor, this);
        _depth_sub  = _n.subscribe("/camera/aligned_depth_to_color/image_raw", 10, &Listener::updateDepth, this);
        // _depth_sub  = _n.subscribe("/camera/depth/image_rect_raw", 10, &Listener::updateDepth, this);

        _local_position_sub = _n.subscribe("/mavros/global_position/local", 10, &Listener::updateLocalPosition, this);
        _rel_alt_sub = _n.subscribe("/mavros/global_position/rel_alt", 10, &Listener::updateAlt, this);
        _vel_body_sub = _n.subscribe("/mavros/local_position/velocity_body", 10, &Listener::updateVelBody, this);
        _rc_in_sub = _n.subscribe("/mavros/rc/in", 10, &Listener::updateRCIn, this);
        _state_sub = _n.subscribe("/mavros/state", 10, &Listener::updateState, this);

    }


    /** Main run function**/
    void run()
    {
        ros::Timer timer = _n.createTimer(ros::Duration(1.0/_loop_rate), &Listener::iteration, this);
        ros::spin();

        _bag.close();
    }


    private:

    void iteration(const ros::TimerEvent&)
    {
        _telemetry.header.seq += 1;
        _telemetry.header.stamp = ros::Time::now();
        _bag.write("/my_telemetry", ros::Time::now(), _telemetry);

        _color_image.header.seq = _telemetry.header.seq;
        _depth_image.header.seq = _telemetry.header.seq;
        _bag.write("/camera/color/image_raw", ros::Time::now(), _color_image);
        _bag.write("/camera/aligned_depth_to_color/image_raw", ros::Time::now(), _depth_image);


    }

    /*       */
    // Color Image
    void updateColor(const sensor_msgs::Image::ConstPtr& msg)
    {          
        _color_image = *msg;
    }

    // Depth Image
    void updateDepth(const sensor_msgs::Image::ConstPtr& msg)
    {          
        _depth_image = *msg;
    }

    // Local Position
    void updateLocalPosition(const nav_msgs::Odometry::ConstPtr& msg)
    {   
        _telemetry.pose = msg->pose;
        _telemetry.twist = msg->twist;

    }

    // Relative Altitude
    void updateAlt(const std_msgs::Float64::ConstPtr& msg)
    {
        _telemetry.rel_alt = msg->data;

    }

    // Body Velocity
    void updateVelBody(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        _telemetry.vel_twist = msg->twist;
    }

    // RC Input
    void updateRCIn(const mavros_msgs::RCIn::ConstPtr& msg)
    {
        _telemetry.channels = msg->channels;
    }

    // State
    void updateState(const mavros_msgs::State::ConstPtr& msg)
    {
        _telemetry.mode = msg->mode;
    }


    private:

    double _loop_rate;

    ros::NodeHandle _n;
    rosbag::Bag _bag;
    ros::Subscriber _color_sub;
    ros::Subscriber _depth_sub;
    ros::Subscriber _local_position_sub;
    ros::Subscriber _rel_alt_sub;
    ros::Subscriber _vel_body_sub;
    ros::Subscriber _rc_in_sub;
    ros::Subscriber _state_sub;

    sensor_msgs::Image _color_image;
    sensor_msgs::Image _depth_image;

    neptune::Custom _telemetry;

};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;

    // loop rate (default 10Hz)
    double rate;
    n.param("rate", rate, 10.0);
    ros::Rate loop_rate(rate);
   
    // bag file definition
    std::time_t now = time(0);
    struct tm * timeinfo = localtime(&(now));
    char buffer [30];
    strftime(buffer,30,"%Y_%h_%d_%H_%M_%S.bag", timeinfo);
    //char filename[50] = "/media/peng/Samsung/";
    char filename[50] = "/home/lab/Documents/";
    std::strcat(filename, buffer);

    Listener listener(filename, rate);
    ros::Duration(1.0).sleep(); // let Listener update its internal states
    listener.run();
}