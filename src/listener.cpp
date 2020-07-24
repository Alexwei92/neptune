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

        _color_sub  = _n.subscribe("/camera/color/image_raw/compressed", 10, &Listener::updateColor, this);
        _depth_sub  = _n.subscribe("/camera/depth/image_rect_raw/compressed", 10, &Listener::updateDepth, this);
        _infra1_sub = _n.subscribe("/camera/infra1/image_rect_raw/compressed", 10, &Listener::updateInfra1, this);
        _infra2_sub = _n.subscribe("/camera/infra2/image_rect_raw/compressed", 10, &Listener::updateInfra2, this);

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
        _bag.write("/camera/color/image_raw/compressed", ros::Time::now(), _color_image_compressed);
        _bag.write("/camera/depth/image_rect_raw/compressed", ros::Time::now(), _depth_image_compressed);
        _bag.write("/camera/infra1/image_rect_aw/compressed", ros::Time::now(), _infra1_image_compressed);
        _bag.write("/camera/infra2/image_rect_raw/compressed", ros::Time::now(), _infra2_image_compressed);
        _bag.write("/mavros/global_position/local", ros::Time::now(), _local_position);
        _bag.write("/mavros/global_position/rel_alt", ros::Time::now(), _rel_alt);
        _bag.write("/mavros/local_position/velocity_body", ros::Time::now(), _vel_body);
        _bag.write("/mavros/rc/in", ros::Time::now(), _rc_in);
        _bag.write("/mavros/state", ros::Time::now(), _state);
    }

    /*       */
    // Color Image
    void updateColor(const sensor_msgs::CompressedImage::ConstPtr& msg)
    {          
        _color_image_compressed = *msg;
    }

    // Depth Image
    void updateDepth(const sensor_msgs::CompressedImage::ConstPtr& msg)
    {          
        _depth_image_compressed = *msg;
    }

    // Infra1 Image
    void updateInfra1(const sensor_msgs::CompressedImage::ConstPtr& msg)
    {          
        _infra1_image_compressed = *msg;
    }

    // Infra2 Image
    void updateInfra2(const sensor_msgs::CompressedImage::ConstPtr& msg)
    {          
        _infra2_image_compressed = *msg;
    }

    // Local Position
    void updateLocalPosition(const nav_msgs::Odometry::ConstPtr& msg)
    {   
        _local_position = *msg;
    }

    // Relative Altitude
    void updateAlt(const std_msgs::Float64::ConstPtr& msg)
    {
        _rel_alt = *msg;
    }

    // Body Velocity
    void updateVelBody(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        _vel_body = *msg;
    }

    // RC Input
    void updateRCIn(const mavros_msgs::RCIn::ConstPtr& msg)
    {
        _rc_in = *msg;
    }

    // State
    void updateState(const mavros_msgs::State::ConstPtr& msg)
    {
        _state = *msg;
    }

    // enum status
    // {
    //     Idle = 0,
    //     Recording = 1,
    // };

    private:

    double _loop_rate;

    ros::NodeHandle _n;
    rosbag::Bag _bag;
    ros::Subscriber _color_sub;
    ros::Subscriber _depth_sub;
    ros::Subscriber _infra1_sub;
    ros::Subscriber _infra2_sub;
    ros::Subscriber _local_position_sub;
    ros::Subscriber _rel_alt_sub;
    ros::Subscriber _vel_body_sub;
    ros::Subscriber _rc_in_sub;
    ros::Subscriber _state_sub;

    sensor_msgs::CompressedImage _color_image_compressed;
    sensor_msgs::CompressedImage _depth_image_compressed;
    sensor_msgs::CompressedImage _infra1_image_compressed;
    sensor_msgs::CompressedImage _infra2_image_compressed;
    nav_msgs::Odometry _local_position;
    geometry_msgs::TwistStamped _vel_body;
    std_msgs::Float64 _rel_alt;
    mavros_msgs::RCIn _rc_in;
    mavros_msgs::State _state;

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
    char filename[50] = "/home/lab/Documents/Peng/bags/";
    std::strcat(filename, buffer);

    Listener listener(filename, rate);
    listener.run();

}

// /mavros/vfr_hud  
