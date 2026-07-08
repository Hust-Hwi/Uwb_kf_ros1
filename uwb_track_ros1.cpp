#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <string>
#include <regex>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <mutex>              
#include <condition_variable> 
#include <Eigen/Dense>

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>

using namespace Eigen;

void normalize_angle(double& angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
}

class KalmanFilter2D {
public:
    double dt;
    Vector4d X;
    Matrix4d P, A, Q;
    Matrix<double, 2, 4> H;
    Matrix2d R;
    bool is_initialized;

    explicit KalmanFilter2D(double dt = 0.1) : dt(dt), is_initialized(false) {
        X = Vector4d::Zero();
        P = Matrix4d::Identity();
        A = Matrix4d::Identity();
        A(0, 2) = dt;
        A(1, 3) = dt;
        H << 1, 0, 0, 0,
             0, 1, 0, 0; 

        R = Matrix2d::Identity() * 0.1; 
        
        double q_pos = 0.1;    
        double q_vel = 1.0;   
        Q = Matrix4d::Zero();
        Q(0, 0) = q_pos * dt;
        Q(1, 1) = q_pos * dt;
        Q(2, 2) = q_vel * dt;
        Q(3, 3) = q_vel * dt;
    }

    void update_dt(double new_dt) {
        if (new_dt > 0) {
            dt = new_dt;
            A(0, 2) = dt;
            A(1, 3) = dt;
            double q_pos = 0.1;    
            double q_vel = 1.0;   
            Q(0, 0) = q_pos * dt;
            Q(1, 1) = q_pos * dt;
            Q(2, 2) = q_vel * dt;
            Q(3, 3) = q_vel * dt;
        }
    }

    void update_R_matrix(double x, double y, double sigma_r = 5.0, double sigma_theta_deg = 3.0, double scale_factor = 1000.0) {
        double r = std::hypot(x, y);
        double theta = std::atan2(y, x);
        double sigma_theta = sigma_theta_deg * M_PI / 180.0;
        double r_var = sigma_r * sigma_r;
        double theta_var_proj = std::pow(r * sigma_theta, 2);
        double cos_th = std::cos(theta);
        double sin_th = std::sin(theta);
        
        R(0, 0) = r_var * std::pow(cos_th, 2) + theta_var_proj * std::pow(sin_th, 2);
        R(1, 1) = r_var * std::pow(sin_th, 2) + theta_var_proj * std::pow(cos_th, 2);
        R(0, 1) = (r_var - theta_var_proj) * sin_th * cos_th;
        R(1, 0) = R(0, 1);
        R *= scale_factor;
    }

    void predict() {
        X = A * X;
        P = A * P * A.transpose() + Q;
    }

    Vector4d update(const Vector2d& z) {
        if (!is_initialized) {
            X(0) = z(0);
            X(1) = z(1);
            X(2) = 0.0; 
            X(3) = 0.0;
            is_initialized = true;
            return X;
        }
        Matrix2d S = H * P * H.transpose() + R;
        Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();
        Vector2d y = z - H * X;
        X = X + K * y;
        P = P - K * H * P;
        return X;
    }
};

class UwbApiClient {
public:
    UwbApiClient(const std::string& base_url,
                 ros::Publisher nav_pub,
                 double max_nav_speed_m_s,
                 double goal_publish_interval_sec,
                 double absolute_tracking_distance,
                 double max_fov_deg,
                 double angle_follow_scale) 
        : base_url_(base_url),
          nav_pub_(nav_pub),
          max_nav_speed_m_s_(max_nav_speed_m_s),
          goal_publish_interval_sec_(goal_publish_interval_sec),
          absolute_tracking_distance_(absolute_tracking_distance),
          max_fov_deg_(max_fov_deg),
          angle_follow_scale_(angle_follow_scale),
          is_running_(true),
          has_latest_goal_(false),
          has_published_goal_(false),
          nav_blocked_(false),
          last_goal_process_time_(0),
          last_goal_publish_time_(0) {
        worker_thread_ = std::thread(&UwbApiClient::networkWorkerLoop, this);
    }

    ~UwbApiClient() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_running_ = false;
        }
        cv_.notify_one();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    void updateLatestGoal(double body_x, double body_y, double body_yaw, double body_distance) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (nav_blocked_) {
                return;
            }
            latest_goal_.stamp = ros::Time::now();
            latest_goal_.body_x = body_x;
            latest_goal_.body_y = body_y;
            latest_goal_.body_yaw = body_yaw;
            latest_goal_.body_distance = body_distance;
            has_latest_goal_ = true; 
        }
        cv_.notify_one(); 
    }

    void setNavBlocked(bool blocked) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (nav_blocked_ == blocked) {
            return;
        }
        nav_blocked_ = blocked;
        if (nav_blocked_) {
            has_latest_goal_ = false;
        }
        cv_.notify_one();
    }

private:
    struct BodyGoal {
        ros::Time stamp;
        double body_x = 0.0;
        double body_y = 0.0;
        double body_yaw = 0.0;
        double body_distance = 0.0;
    };

    struct RobotPose {
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
    };

    void networkWorkerLoop() {
        while (true) {
            if (!waitUntilReadyToProcess()) {
                break;
            }

            BodyGoal goal;
            if (getLatestGoal(goal)) {
                last_goal_process_time_ = goal.stamp;
                if (goal.body_distance < absolute_tracking_distance_) {
                    continue;
                }

                RobotPose robot_pose;
                if (fetchRobotPose(robot_pose)) {
                    processGoal(goal, robot_pose);
                }
            }
        }
    }

    bool waitUntilReadyToProcess() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !is_running_ || has_latest_goal_;
            });

            if (!is_running_) {
                return false;
            }

            if (latest_goal_.stamp == last_goal_process_time_) {
                cv_.wait(lock, [this] {
                    return !is_running_ || !has_latest_goal_ || latest_goal_.stamp != last_goal_process_time_;
                });
                continue;
            }

            if (!has_published_goal_) {
                return true;
            }

            double wait_sec = goal_publish_interval_sec_ - (ros::Time::now() - last_goal_publish_time_).toSec();
            if (wait_sec > 0.0) {
                cv_.wait_for(lock,
                             std::chrono::duration<double>(wait_sec),
                             [this] { return !is_running_ || !has_latest_goal_; });

                if (!is_running_) {
                    return false;
                }
                continue;
            }

            return true;
        }
    }

    bool getLatestGoal(BodyGoal& goal) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_latest_goal_) {
            return false;
        }
        goal = latest_goal_;
        return true;
    }

    bool fetchRobotPose(RobotPose& pose) {
        std::string get_url = base_url_ + "/getLocalization";
        std::string response = "";
        char buffer[128];
        char get_cmd[512];
        snprintf(get_cmd, sizeof(get_cmd),
                 "curl --connect-timeout 1 --max-time 1.5 -s -X GET %s",
                 get_url.c_str());

        FILE* pipe = popen(get_cmd, "r");
        if (pipe) {
            while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
                response += buffer;
            }
            pclose(pipe);
        }

        if (response.empty()) {
            ROS_WARN_THROTTLE(2.0, "获取机器人位姿超时，跳过本周期导航点计算。");
            return false;
        }

        std::regex code_re("\"code\"\\s*:\\s*(\\d+)");
        std::regex pos_re("\"position\"\\s*:\\s*\\[\\s*([-+\\d\\.eE]+)\\s*,\\s*([-+\\d\\.eE]+)");
        std::regex ori_re("\"orientation\"\\s*:\\s*([-+\\d\\.eE]+)");

        std::smatch match;
        int code = -1;
        if (std::regex_search(response, match, code_re)) {
            code = std::stoi(match[1].str());
        } else {
            ROS_WARN_THROTTLE(2.0, "JSON中找不到 code. 原始返回: %s", response.c_str());
            return false;
        }

        if (code != 0) {
            ROS_WARN_THROTTLE(2.0, "获取位姿失败, code=%d. 原始返回: %s", code, response.c_str());
            return false;
        }

        if (std::regex_search(response, match, pos_re)) {
            pose.x = std::stod(match[1].str());
            pose.y = std::stod(match[2].str());
        } else {
            ROS_WARN_THROTTLE(2.0, "JSON中找不到 position. 原始返回: %s", response.c_str());
            return false;
        }

        if (std::regex_search(response, match, ori_re)) {
            pose.yaw = std::stod(match[1].str());
        } else {
            ROS_WARN_THROTTLE(2.0, "JSON中找不到 orientation. 原始返回: %s", response.c_str());
            return false;
        }

        return true;
    }

    void processGoal(const BodyGoal& goal, const RobotPose& robot_pose) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (nav_blocked_) {
                return;
            }
        }

        if (goal.body_distance < absolute_tracking_distance_) {
            return;
        }

        double target_body_yaw = goal.body_yaw;
        double body_yaw_deg = goal.body_yaw * 180.0 / M_PI;
        bool is_angle_out_of_bounds = std::abs(body_yaw_deg) > max_fov_deg_;
        if (is_angle_out_of_bounds) {
            target_body_yaw *= angle_follow_scale_;
        }

        double target_world_x = robot_pose.x + goal.body_x * std::cos(robot_pose.yaw) - goal.body_y * std::sin(robot_pose.yaw);
        double target_world_y = robot_pose.y + goal.body_x * std::sin(robot_pose.yaw) + goal.body_y * std::cos(robot_pose.yaw);
        double target_world_yaw = robot_pose.yaw + target_body_yaw; 
        normalize_angle(target_world_yaw);

        ros::Time now = ros::Time::now();
        if (has_published_goal_) {
            double publish_elapsed = (now - last_goal_publish_time_).toSec();
            if (publish_elapsed < goal_publish_interval_sec_) {
                return;
            }

            double dt_goal = publish_elapsed;
            if (dt_goal > 0.0) {
                double dist_to_last_goal = std::hypot(target_world_x - last_goal_world_x_, target_world_y - last_goal_world_y_);
                double goal_speed = dist_to_last_goal / dt_goal;
                if (goal_speed > max_nav_speed_m_s_) {
                    ROS_WARN_THROTTLE(1.0, "map导航点跳变过快 (%.2f m/s)，超限拦截！", goal_speed);
                    return;
                }
            }
        }

        if (is_angle_out_of_bounds) {
            //由于D1-O1+D1-O2转向超调过大，所以只能转一半角度
            ROS_INFO_THROTTLE(1.0, "目标偏角过大 (%.1f 度)，触发转向跟随！", body_yaw_deg);
        }

        if (publishAndSendGoal(target_world_x, target_world_y, target_world_yaw)) {
            last_goal_world_x_ = target_world_x;
            last_goal_world_y_ = target_world_y;
            last_goal_publish_time_ = now;
            has_published_goal_ = true;
        }
    }

    bool publishAndSendGoal(double world_x, double world_y, double final_yaw) {
        std::string post_url = base_url_ + "/setGoals";
        std::string goto_url = base_url_ + "/goto";

        geometry_msgs::PoseStamped nav_msg;
        nav_msg.header.stamp = ros::Time::now();
        nav_msg.header.frame_id = "map"; 
        nav_msg.pose.position.x = world_x;
        nav_msg.pose.position.y = world_y;
        nav_msg.pose.orientation.z = std::sin(final_yaw / 2.0);
        nav_msg.pose.orientation.w = std::cos(final_yaw / 2.0);
        nav_pub_.publish(nav_msg);

        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "curl --connect-timeout 1 --max-time 1.5 -s -X POST %s -H \"Content-Type: application/json\" "
                 "-d '{\"goal_list\": [{\"x\": %f, \"y\": %f, \"theta\": %f}]}' > /dev/null",
                 post_url.c_str(), world_x, world_y, final_yaw);
        
        if (std::system(cmd) == 0) {
            char goto_cmd[256];
            snprintf(goto_cmd, sizeof(goto_cmd),
                     "curl --connect-timeout 1 --max-time 1.5 -s -X POST %s -H \"Content-Type: application/json\" "
                     "-d '{\"goal_id\": 0}' > /dev/null",
                     goto_url.c_str());
            std::system(goto_cmd);
            
            ROS_INFO_THROTTLE(1.0, "成功同步最新追踪点至全局世界地图: X=%.2f, Y=%.2f, YAW=%.2f", world_x, world_y, final_yaw);
            return true;
        }

        ROS_WARN_THROTTLE(2.0, "执行 setGoals 失败！");
        return false;
    }

    std::string base_url_;
    ros::Publisher nav_pub_;
    double max_nav_speed_m_s_;
    double goal_publish_interval_sec_;
    double absolute_tracking_distance_;
    double max_fov_deg_;
    double angle_follow_scale_;

    // 并发控制成员变量
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool is_running_;
    bool has_latest_goal_;
    bool nav_blocked_;

    BodyGoal latest_goal_;

    // map坐标系下的上一条已下发目标，用于世界系速度/距离判断
    bool has_published_goal_;
    ros::Time last_goal_process_time_;
    ros::Time last_goal_publish_time_;
    double last_goal_world_x_ = 0.0;
    double last_goal_world_y_ = 0.0;

};

class UwbTrackNode {
public:
    explicit UwbTrackNode(ros::NodeHandle& nh) : nh_(nh) {
        raw_pub_ = nh_.advertise<geometry_msgs::PointStamped>("uwb/raw_pose", 10);
        gated_pub_ = nh_.advertise<geometry_msgs::PointStamped>("uwb/gated_pose", 10);
        filtered_pub_ = nh_.advertise<geometry_msgs::PointStamped>("uwb/filtered_pose", 10);
        nav_goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("uwb/nav_goal", 10);
        raw_dist_angle_pub_ = nh_.advertise<geometry_msgs::PointStamped>("uwb/raw_dist_angle", 10);

        api_client_ = std::make_shared<UwbApiClient>(
            "http://192.168.121.1:10000",
            nav_goal_pub_,
            MAX_NAV_SPEED_M_S,
            GOAL_PUBLISH_INTERVAL_SEC,
            ABSOLUTE_TRACKING_DISTANCE,
            MAX_FOV_DEG,
            ANGLE_FOLLOW_SCALE);
        
        serial_fd_ = init_serial("/dev/uwb");
        if (serial_fd_ < 0) {
            ROS_ERROR("无法打开串口 /dev/uwb");
            throw std::runtime_error("Serial port error");
        }

        ROS_INFO("UWB Track 节点已成功启动。");
        serial_thread_ = std::thread(&UwbTrackNode::serial_read_loop, this);
    }

    ~UwbTrackNode() {
        if (serial_fd_ >= 0) close(serial_fd_);
        if (serial_thread_.joinable()) serial_thread_.join();
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher raw_pub_;
    ros::Publisher gated_pub_;
    ros::Publisher filtered_pub_;
    ros::Publisher nav_goal_pub_;
    ros::Publisher raw_dist_angle_pub_;
    int serial_fd_;
    std::thread serial_thread_;
    std::shared_ptr<UwbApiClient> api_client_;
    
    // 追踪目标位置的状态变量
    const double FOLLOW_DISTANCE = 1;             // 减去的跟随距离
    const double ABSOLUTE_TRACKING_DISTANCE = 1;  // 目标绝对距离大于此阈值时触发追踪
    const double GOAL_PUBLISH_INTERVAL_SEC = 0.33333333; // 每间隔多少秒发布一次导航点
    const double MAX_NAV_SPEED_M_S = 3.0;         // 导航点的最快速度
    const double MAX_FOV_DEG = 40.0;              // 偏角超过该值时缩小 yaw 跟随量
    const double ANGLE_FOLLOW_SCALE = 0.5;        // 角度过大时只跟随一半，降低超调

    int init_serial(const char* portname) {
        int fd = open(portname, O_RDWR | O_NOCTTY | O_SYNC);
        if (fd < 0) return -1;
        struct termios tty;
        if (tcgetattr(fd, &tty) != 0) return -1;
        cfsetospeed(&tty, B115200); cfsetispeed(&tty, B115200);
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_iflag &= ~IGNBRK; tty.c_lflag = 0; tty.c_oflag = 0;
        tty.c_cc[VMIN] = 1; tty.c_cc[VTIME] = 1;
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~(PARENB | PARODD);
        tty.c_cflag &= ~CSTOPB; tty.c_cflag &= ~CRTSCTS;
        if (tcsetattr(fd, TCSANOW, &tty) != 0) return -1;
        return fd;
    }

    void serial_read_loop() {
        KalmanFilter2D kf(0.1);
        
        double last_valid_r = 0.0, last_valid_theta = 0.0;
        double last_valid_x = 0.0, last_valid_y = 0.0; //有效点XY坐标
        auto last_valid_time = ros::Time::now();
        int consecutive_rejects = 0;
        const int MAX_REJECT_COUNT = 60;

        const double MAX_SPEED_R_M_S = 3.0;            // 径向距离最大跳变速度
        const double MAX_SPEED_THETA_RAD_S = 1.5;      // 角度最大变换速度
        const double MAX_SPEED_XY_M_S = 3.0;           // 二维平面最大跳变速度
        const double MIN_RADIUS_FOR_ANGLE_CHECK = 0.5; // 小于 0.5m 时豁免角度跳变检查
        const double KF_SIGMA_R_M = 0.2;
        const double KF_SIGMA_THETA_DEG = 12.0;         // 角度测量权重
        const double KF_R_SCALE = 20.0;
        
        ros::Time last_reject_time = ros::Time(0);
        const double NAV_BLOCK_DURATION = 0.25;  // 拒绝后导航静默期，避免坏点刚出现时仍下发跳变目标

        bool has_smoothed_nav_y = false;
        double smoothed_nav_y = 0.0;
        ros::Time last_smoothed_nav_y_time = ros::Time(0);
        int stable_nav_y_sign = 0;
        int pending_nav_y_sign = 0;
        int pending_nav_y_sign_count = 0;

        const double BODY_Y_DEADZONE_ENTER_M = 0.12;
        const double BODY_Y_DEADZONE_EXIT_M = 0.25;
        const double MAX_BODY_Y_SPEED_M_S = 0.35;
        const int BODY_Y_SIGN_CONFIRM_COUNT = 4;

        auto sign_no_zero = [](double value) -> int {
            return value >= 0.0 ? 1 : -1;
        };

        auto limit_step = [](double current, double target, double max_step) -> double {
            double delta = target - current;
            if (delta > max_step) return current + max_step;
            if (delta < -max_step) return current - max_step;
            return target;
        };

        auto last_time = ros::Time::now();
        uint8_t buf[256];
        int buf_len = 0;

        while (ros::ok()) {
            uint8_t temp_buf[64];
            int n = read(serial_fd_, temp_buf, sizeof(temp_buf));
            if (n > 0) {
                if (buf_len + n > static_cast<int>(sizeof(buf))) {
                    ROS_WARN_THROTTLE(1.0, "串口缓存即将溢出，丢弃当前缓存以避免越界");
                    buf_len = 0;
                }
                memcpy(buf + buf_len, temp_buf, n);
                buf_len += n;
                int header_idx = -1;
                for (int i = 0; i <= buf_len - 4; ++i) {
                    if (buf[i]==0xFF && buf[i+1]==0xFF && buf[i+2]==0xFF && buf[i+3]==0xFF) {
                        header_idx = i; break;
                    }
                }
                if (header_idx != -1) {
                    if (header_idx > 0) {
                        memmove(buf, buf + header_idx, buf_len - header_idx);
                        buf_len -= header_idx;
                    }
                    if (buf_len >= 6) {
                        uint16_t packet_length = (buf[4] << 8) | buf[5];
                        if (packet_length < 26 || packet_length > sizeof(buf)) {
                            ROS_WARN_THROTTLE(1.0, "收到非法包长 %u，丢弃当前缓存", packet_length);
                            buf_len = 0;
                            continue;
                        }
                        if (buf_len >= packet_length) {
                            uint16_t command = (buf[8] << 8) | buf[9];
                            if (command == 0x2001) {
                                uint32_t distance = (buf[20] << 24) | (buf[21] << 16) | (buf[22] << 8) | buf[23];
                                int16_t azimuth = (buf[24] << 8) | buf[25];
                                auto current_time = ros::Time::now();
                                
                                geometry_msgs::PointStamped msg_raw_da;
                                msg_raw_da.header.stamp = current_time;
                                msg_raw_da.header.frame_id = "uwb_base_link";
                                msg_raw_da.point.x = distance; 
                                msg_raw_da.point.y = azimuth;  
                                raw_dist_angle_pub_.publish(msg_raw_da);

                                double raw_r = distance / 100.0; // cm 转 m
                                double raw_theta = azimuth * M_PI / 180.0;
                                normalize_angle(raw_theta);

                                double raw_x = raw_r * std::cos(raw_theta);
                                double raw_y = raw_r * std::sin(raw_theta);
                                
                                geometry_msgs::PointStamped msg_raw;
                                msg_raw.header.stamp = current_time;
                                msg_raw.header.frame_id = "uwb_base_link";
                                msg_raw.point.x = raw_x; 
                                msg_raw.point.y = raw_y;
                                raw_pub_.publish(msg_raw);

                                double dt = (current_time - last_time).toSec();
                                last_time = current_time;
                                kf.update_dt(dt); 
                                kf.predict(); 

                                bool is_valid = false;

                                if (kf.is_initialized) {
                                    double dt_valid = (current_time - last_valid_time).toSec();
                                    if (dt_valid > 0) {
                                        double valid_speed_r = std::abs(raw_r - last_valid_r) / dt_valid;

                                        double dist_theta = raw_theta - last_valid_theta;
                                        normalize_angle(dist_theta);
                                        double valid_speed_theta = std::abs(dist_theta) / dt_valid;

                                        double valid_speed_xy = std::hypot(raw_x - last_valid_x, raw_y - last_valid_y) / dt_valid;

                                        bool r_valid = (valid_speed_r <= MAX_SPEED_R_M_S);
                                        bool theta_valid = (raw_r <= MIN_RADIUS_FOR_ANGLE_CHECK) || (valid_speed_theta <= MAX_SPEED_THETA_RAD_S);
                                        bool xy_valid = (valid_speed_xy <= MAX_SPEED_XY_M_S);

                                        if (r_valid && theta_valid && xy_valid) {
                                            is_valid = true;
                                        }
                                    }
                                } else {
                                    is_valid = true;
                                }

                                if (!is_valid) {
                                    if (kf.is_initialized) {
                                        consecutive_rejects++;
                                        last_reject_time = current_time; 
                                        if (api_client_) {
                                            api_client_->setNavBlocked(true);
                                        }
                                        has_smoothed_nav_y = false;
                                        smoothed_nav_y = 0.0;
                                        stable_nav_y_sign = 0;
                                        pending_nav_y_sign = 0;
                                        pending_nav_y_sign_count = 0;
                                        
                                        if (consecutive_rejects >= MAX_REJECT_COUNT) {
                                            kf.is_initialized = false;
                                            consecutive_rejects = 0;
                                            is_valid = true; 
                                        }
                                    }
                                } else {
                                    consecutive_rejects = 0;
                                }

                                if (is_valid) {
                                    last_valid_r = raw_r; 
                                    last_valid_theta = raw_theta;
                                    last_valid_x = raw_x; 
                                    last_valid_y = raw_y; 
                                    last_valid_time = current_time; 
                                    
                                    geometry_msgs::PointStamped msg_gated = msg_raw;
                                    gated_pub_.publish(msg_gated);
                                    
                                    kf.update_R_matrix(raw_x, raw_y, KF_SIGMA_R_M, KF_SIGMA_THETA_DEG, KF_R_SCALE);
                                    Vector2d z(raw_x, raw_y); 
                                    Vector4d state = kf.update(z);
                                    
                                    double filtered_x_m = state(0);
                                    double filtered_y_m = state(1);
                                    
                                    geometry_msgs::PointStamped msg_filt;
                                    msg_filt.header.stamp = current_time;
                                    msg_filt.header.frame_id = "uwb_base_link";
                                    msg_filt.point.x = filtered_x_m; 
                                    msg_filt.point.y = filtered_y_m; 
                                    filtered_pub_.publish(msg_filt);

                                    double nav_target_x = filtered_x_m;
                                    double nav_target_y = filtered_y_m;
                                    double d = std::hypot(nav_target_x, nav_target_y);

                                    if (d > 1e-6) {
                                        double goal_distance = std::max(0.0, d - FOLLOW_DISTANCE);
                                        double scale = goal_distance / d;
                                        nav_target_x *= scale;
                                        nav_target_y *= scale;
                                    } else {
                                        nav_target_x = 0.0;
                                        nav_target_y = 0.0;
                                    }

                                    bool is_nav_blocked = false;
                                    if (last_reject_time.toSec() != 0) {
                                        is_nav_blocked = (current_time - last_reject_time).toSec() < NAV_BLOCK_DURATION;
                                    }

                                    if (api_client_) {
                                        api_client_->setNavBlocked(is_nav_blocked);
                                        if (is_nav_blocked) {
                                            ROS_WARN_THROTTLE(1.0, "UWB信号抖动/被遮挡(存在近期丢弃点)，暂停导航点下发。");
                                            has_smoothed_nav_y = false;
                                            smoothed_nav_y = 0.0;
                                            stable_nav_y_sign = 0;
                                            pending_nav_y_sign = 0;
                                            pending_nav_y_sign_count = 0;
                                        } else {

                                            double target_nav_y = nav_target_y;
                                            double target_nav_y_abs = std::abs(target_nav_y);
                                            int candidate_nav_y_sign = 0;

                                            if (stable_nav_y_sign == 0) {
                                                if (target_nav_y_abs < BODY_Y_DEADZONE_EXIT_M) {
                                                    target_nav_y = 0.0;
                                                } else {
                                                    candidate_nav_y_sign = sign_no_zero(target_nav_y);
                                                }
                                            } else {
                                                if (target_nav_y_abs < BODY_Y_DEADZONE_ENTER_M) {
                                                    target_nav_y = 0.0;
                                                } else {
                                                    candidate_nav_y_sign = sign_no_zero(target_nav_y);
                                                }
                                            }

                                            if (candidate_nav_y_sign != 0 &&
                                                stable_nav_y_sign != 0 &&
                                                candidate_nav_y_sign != stable_nav_y_sign) {
                                                if (pending_nav_y_sign == candidate_nav_y_sign) {
                                                    pending_nav_y_sign_count++;
                                                } else {
                                                    pending_nav_y_sign = candidate_nav_y_sign;
                                                    pending_nav_y_sign_count = 1;
                                                }

                                                if (pending_nav_y_sign_count < BODY_Y_SIGN_CONFIRM_COUNT) {
                                                    target_nav_y = smoothed_nav_y;
                                                } else {
                                                    stable_nav_y_sign = candidate_nav_y_sign;
                                                    pending_nav_y_sign = 0;
                                                    pending_nav_y_sign_count = 0;
                                                }
                                            } else {
                                                stable_nav_y_sign = candidate_nav_y_sign;
                                                pending_nav_y_sign = 0;
                                                pending_nav_y_sign_count = 0;
                                            }

                                            double smooth_dt = GOAL_PUBLISH_INTERVAL_SEC;
                                            if (has_smoothed_nav_y && last_smoothed_nav_y_time.toSec() != 0.0) {
                                                smooth_dt = (current_time - last_smoothed_nav_y_time).toSec();
                                                if (smooth_dt <= 0.0 || smooth_dt > 1.0) {
                                                    smooth_dt = GOAL_PUBLISH_INTERVAL_SEC;
                                                }
                                            } else {
                                                smoothed_nav_y = 0.0;
                                                has_smoothed_nav_y = true;
                                            }

                                            double max_nav_y_step = MAX_BODY_Y_SPEED_M_S * smooth_dt;
                                            smoothed_nav_y = limit_step(smoothed_nav_y, target_nav_y, max_nav_y_step);
                                            if (target_nav_y == 0.0 && std::abs(smoothed_nav_y) < BODY_Y_DEADZONE_ENTER_M) {
                                                smoothed_nav_y = 0.0;
                                            }

                                            last_smoothed_nav_y_time = current_time;
                                            nav_target_y = smoothed_nav_y;
                                            double nav_target_yaw = std::atan2(nav_target_y, nav_target_x);

                                            api_client_->updateLatestGoal(nav_target_x, nav_target_y, nav_target_yaw, d);
                                        }
                                    }
                                }
                            }
                            memmove(buf, buf + packet_length, buf_len - packet_length); buf_len -= packet_length;
                        }
                    }
                } else if (buf_len > 200) buf_len = 0;
            }
        }
    }
};

int main(int argc, char * argv[]) {
    ros::init(argc, argv, "uwb_track_node");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");
    UwbTrackNode node(nh);
    ros::spin();
    return 0;
}
