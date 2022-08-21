#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    auto rot_rad = rotation_angle / 180.0f * CV_PI;
    auto c = cos(rot_rad);
    auto s = sin(rot_rad);
    model(0, 0) = c;
    model(0, 1) = -s;
    model(1, 0) = s;
    model(1, 1) = c;
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    zFar = -zFar;
    zNear = -zNear;
    Eigen::Matrix4f persp = Eigen::Matrix4f::Zero();
    persp(0, 0) = zNear;
    persp(1, 1) = zNear;
    persp(2, 2) = zNear + zFar;
    persp(2, 3) = -zNear * zFar;
    persp(3, 2) = 1;
    Eigen::Matrix4f ortho = Eigen::Matrix4f::Zero();
    auto h = abs(zNear) * tan(eye_fov / 180.0f * CV_PI / 2);
    auto w = aspect_ratio * h;
    auto z_mid = (zNear+zFar)/2;
    auto z_len = abs(zFar-zNear) * 0.5f;
    ortho(0, 0) = 1.0 / w;
    ortho(1, 1) = 1.0 / h;
    ortho(2, 2) = 1.0/z_len;
    ortho(2, 3) = -z_mid/z_len;
    ortho(3, 3) = 1;
    projection = ortho * persp;
    return projection;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle)
{
    axis.normalize();
    auto rad = angle/180.0f*MY_PI;
    Eigen::Matrix3f axis_x;
    axis_x <<        0,  -axis.z(),    axis.y(),
              axis.z(),          0,   -axis.x(), 
             -axis.y(),   axis.x(),           0;
    Eigen::Matrix3f R = cos(rad) * Eigen::Matrix3f::Identity() + (1-cos(rad))*axis*axis.transpose() + sin(rad) * axis_x;
    Eigen::Matrix4f T = Matrix4f::Identity();
    T.block<3, 3>(0, 0) = R;
    return T;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
