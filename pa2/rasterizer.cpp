// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector3f ab(_v[1].x()-_v[0].x(), _v[1].y()-_v[0].y(), 0.0f);
    Eigen::Vector3f bc(_v[2].x()-_v[1].x(), _v[2].y()-_v[1].y(), 0.0f);
    Eigen::Vector3f ca(_v[0].x()-_v[2].x(), _v[0].y()-_v[2].y(), 0.0f);
    Eigen::Vector3f ap(x-_v[0].x(), y-_v[0].y(), 0.0f), bp(x-_v[1].x(), y-_v[1].y(), 0.0f), cp(x-_v[2].x(), y-_v[2].y(), 0.0f);
    auto t1 = ap.cross(ab);
    auto t2 = bp.cross(bc);
    auto t3 = cp.cross(ca);
    return t1.z() * t2.z() > 0 && t1.z() * t3.z() > 0;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = -vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    Eigen::Vector3f v3[3] = {
        Eigen::Vector3f(v[0].x(), v[0].y(), v[0].z()),
        Eigen::Vector3f(v[1].x(), v[1].y(), v[1].z()),
        Eigen::Vector3f(v[2].x(), v[2].y(), v[2].z())
    };
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
    float x_min = width-1, y_min = height-1, x_max = 0, y_max = 0;
    for (const auto& vertex: v)
    {
        x_min = std::min(vertex.x(), x_min);
        x_max = std::max(vertex.x(), x_max);
        y_min = std::min(vertex.y(), y_min);
        y_max = std::max(vertex.y(), y_max);
    }
    int x_min_int = std::max(int(x_min), 0), y_min_int = std::max(int(y_min), 0);
    int x_max_int = std::min(width, int(x_max)), y_max_int = std::min(height, int(y_max));
    float step = 0.25f;
    for (int x=x_min_int;x<x_max_int;++x)
    {
        for (int y=y_min_int;y<y_max_int;++y)
        {
            // float cen_x = float(x) + 0.5f;
            // float cen_y = float(y) + 0.5f;
            // if (!insideTriangle(cen_x, cen_y, v3))
            //     continue;
            // auto[alpha, beta, gamma] = computeBarycentric2D(cen_x, cen_y, t.v);
            // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            // z_interpolated *= w_reciprocal;
            // auto ind = get_index(x, y);
            // if (z_interpolated < depth_buf[ind])
            // {
            //     depth_buf[ind] = z_interpolated;
            //     frame_buf[ind] = t.getColor();
            // }
            auto ind = get_index(x, y);
            auto s_ind_base = ind << 2;
            auto x_f = float(x), y_f = float(y);
            float s_x[4] = {x_f+step, x_f+step, x_f+3*step, x_f+3*step};
            float s_y[4] = {y_f+step, y_f+3*step, y_f+step, y_f+3*step};
            bool update_flag = false;
            for(int i=0;i<4;++i)
            {
                auto cen_x = s_x[i], cen_y=s_y[i];
                if (!insideTriangle(cen_x, cen_y, v3))
                    continue;
                auto[alpha, beta, gamma] = computeBarycentric2D(cen_x, cen_y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                auto s_ind = s_ind_base + i;
                if (z_interpolated < depth_buf_4x[s_ind])
                {
                    update_flag = true;
                    depth_buf_4x[s_ind] = z_interpolated;
                    frame_buf_4x[s_ind] = t.getColor();
                }
            }
            if (update_flag)
                frame_buf[ind] = (frame_buf_4x[s_ind_base] + frame_buf_4x[s_ind_base+1] + frame_buf_4x[s_ind_base+2] + frame_buf_4x[s_ind_base+3]) / 4;
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(frame_buf_4x.begin(), frame_buf_4x.end(), Eigen::Vector3f{0,0,0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf_4x.begin(), depth_buf_4x.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    auto sz = frame_buf.size();
    frame_buf_4x.resize(4*sz);
    depth_buf_4x.resize(4*sz);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on