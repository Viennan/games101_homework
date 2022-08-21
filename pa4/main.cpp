#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    auto max_dis = std::sqrt(window.rows * window.rows + window.cols * window.cols);
    auto dis_map = cv::Mat(window.rows, window.cols, CV_32FC1, cv::Scalar(max_dis));

    double half_width = 3;

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        int minc = std::max<int>(0, point.x-half_width), maxc = std::min<int>(window.cols-1, point.x + half_width);
        int minr = std::max<int>(0, point.y-half_width), maxr = std::min<int>(window.rows-1, point.y + half_width);
        for (int r=minr;r<=maxr;++r)
        {
            for (int c=minc;c<=maxc;++c)
            {
                auto dis = std::sqrt(std::pow(point.y-r, 2) + std::pow(point.x-c, 2));
                if (dis < dis_map.at<float>(r, c) && dis < half_width)
                {
                    dis_map.at<float>(r, c) = dis;
                    auto color = 255 * ((half_width-dis) / half_width);
                    if (window.at<cv::Vec3b>(r, c)[0] < color)
                    {
                        window.at<cv::Vec3b>(r, c) = cv::Vec3b(color, color, color);
                    }     
                }
            }
        }
        // window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    if (control_points.size() <= 1)
        return control_points.front();

    auto next_order = control_points.size() - 1;
    auto next_pts = std::vector<cv::Point2f>(next_order);
    for (auto i=0;i<next_order;++i)
    {
        next_pts[i] = t * (control_points[i+1] - control_points[i]) + control_points[i];
    }
    return recursive_bezier(next_pts, t);

}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = recursive_bezier(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            naive_bezier(control_points, window);
            // bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

    return 0;
}
