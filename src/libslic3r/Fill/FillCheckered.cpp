#include "../ClipperUtils.hpp"
#include "../ShortestPath.hpp"
#include "../Surface.hpp"
#include <cmath>
#include <fstream>

#include "FillCheckered.hpp"
#include "libslic3r/Format/OBJ.hpp"
#include "libslic3r/Model.hpp"
#include <Eigen/Dense>
#include <iostream>

// Set to 1 to write debug SVG and optional texture overlay for checkered infill segments
#ifndef FILL_CHECKERED_DEBUG
#define FILL_CHECKERED_DEBUG 0
#endif
#if FILL_CHECKERED_DEBUG
#include "PNGReadWrite.hpp"
#include <boost/filesystem.hpp>
#endif

// static const double PI = 3.14159265358979323846;

namespace Slic3r
{
  inline double to_mm(coord_t v)
  {
    auto mm = scale_(1);
    return double(v) / mm;
  }

  // Forward declaration: defined later in this file
  bool get_uv_for_point(const Model &model, ObjInfo object_info, const Vec3f &P, Eigen::Vector2f &out_uv);

  // Write a single contour (polygon) to an SVG file
  void write_contour_to_svg(const std::vector<Slic3r::Point> &contour,
                            const std::string &filename,
                            int width = 200, int height = 200)
  {
    std::ofstream svg(filename);
    if (!svg.is_open())
      return;

    svg << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
        << "width=\"" << width << "\" height=\"" << height << "\">\n";

    // Build the polygon points string
    svg << "<polygon points=\"";
    for (const auto &pt : contour)
    {
      // convert from integer coordinates to SVG space
      double x = to_mm(pt.x()) + 100;
      double y = to_mm(pt.y()) + 100;
      svg << x << "," << y << " ";
    }
    svg << "\" style=\"fill:none;stroke:black;stroke-width:1\" />\n";

    svg << "</svg>\n";
    svg.close();
  }

  // Write a single contour (polygon) to an SVG file
  void write_points_to_svg(const std::vector<Slic3r::Point> &contour,
                            const std::string &filename)
  {
    std::ofstream svg(filename);
    if (!svg.is_open())
      return;

    auto min_x = 0;
    auto min_y = 0;

    auto max_x = 0;
    auto max_y = 0;

    for (const auto &pt : contour)
    {
      if (pt.x() < min_x)
        min_x = pt.x();
      if (pt.y() < min_y)
        min_y = pt.y();

      if (pt.x() > max_x)
        max_x = pt.x();

      if (pt.y() > max_y)
        max_y = pt.y();
    }

    min_x = std::min(0, min_x);
    min_x = std::abs(min_x);

    min_y = std::min(0, min_y);
    min_y = std::abs(min_y);

    auto width = to_mm(max_x + min_x) + 200;
    auto height = to_mm(max_y + min_y) + 200;

    svg << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
        << "width=\"" << width << "\" height=\"" << height << "\">\n";

    // Build the polygon points string
    svg << "<polygon points=\"";
    for (const auto &pt : contour)
    {
      // convert from integer coordinates to SVG space
      double x = to_mm(pt.x() + min_x) + 100;
      double y = to_mm(pt.y() + min_y) + 100;
      svg << x << "," << y << " ";
    }
    svg << "\" style=\"fill:none;stroke:black;stroke-width:1\" />\n";

    svg << "</svg>\n";
    svg.close();
  }

  // Write black-cell segments (polylines) to SVG in XY. Optional: draw full contour in grey first.
  void write_polylines_to_svg(const Polylines &segments, const std::string &filename,
                             const Polygon *full_contour = nullptr)
  {
    std::ofstream svg(filename);
    if (!svg.is_open())
      return;

    coord_t min_x = 0, min_y = 0, max_x = 0, max_y = 0;
    auto update_bbox = [&](const Point &p) {
      if (p.x() < min_x) min_x = p.x();
      if (p.y() < min_y) min_y = p.y();
      if (p.x() > max_x) max_x = p.x();
      if (p.y() > max_y) max_y = p.y();
    };
    if (full_contour)
      for (const auto &p : full_contour->points)
        update_bbox(p);
    for (const auto &pl : segments)
      for (const auto &p : pl.points)
        update_bbox(p);
    min_x = std::min(coord_t(0), min_x);
    min_y = std::min(coord_t(0), min_y);
    double w = to_mm(max_x - min_x + scale_(20));
    double h = to_mm(max_y - min_y + scale_(20));
    double ox = to_mm(min_x) - 10;
    double oy = to_mm(min_y) - 10;

    svg << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
        << "width=\"" << w << "\" height=\"" << h << "\" "
        << "viewBox=\"" << ox << " " << oy << " " << w << " " << h << "\">\n";
    if (full_contour)
    {
      svg << "<polygon points=\"";
      for (const auto &p : full_contour->points)
        svg << to_mm(p.x()) << "," << to_mm(p.y()) << " ";
      svg << "\" style=\"fill:none;stroke:#888;stroke-width:0.5\" />\n";
    }
    for (const auto &pl : segments)
    {
      if (pl.points.empty()) continue;
      svg << "<polyline points=\"";
      for (const auto &p : pl.points)
        svg << to_mm(p.x()) << "," << to_mm(p.y()) << " ";
      svg << "\" style=\"fill:none;stroke:blue;stroke-width:1\" />\n";
    }
    svg << "</svg>\n";
    svg.close();
  }

  // Write segments in UV space [0,1]x[0,1] to SVG (e.g. to check alignment with checkerboard).
  void write_uv_polylines_to_svg(const Polylines &segments, const Model &model, const ObjInfo &object_info,
                                  coordf_t z_height, const std::string &filename, float cell_size = 1.0f / 64.0f)
  {
    std::ofstream svg(filename);
    if (!svg.is_open())
      return;

    const double size = 400;
    auto uv_to_svg = [size](float u, float v) {
      return std::make_pair(u * size, (1.f - v) * size);
    };
    svg << "<?xml version=\"1.0\" standalone=\"no\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
        << "width=\"" << size << "\" height=\"" << size << "\">\n";
    int cells = (int)(1.f / cell_size + 0.5f);
    if (cells > 0)
    {
      for (int i = 0; i <= cells; ++i)
      {
        double x = i * cell_size * size;
        svg << "<line x1=\"" << x << "\" y1=\"0\" x2=\"" << x << "\" y2=\"" << size << "\" stroke=\"#eee\" stroke-width=\"0.5\"/>\n";
        double y = i * cell_size * size;
        svg << "<line x1=\"0\" y1=\"" << y << "\" x2=\"" << size << "\" y2=\"" << y << "\" stroke=\"#eee\" stroke-width=\"0.5\"/>\n";
      }
    }
    for (const auto &pl : segments)
    {
      if (pl.points.empty()) continue;
      Eigen::Vector2f uv;
      bool any = false;
      svg << "<polyline points=\"";
      for (const auto &p : pl.points)
      {
        if (get_uv_for_point(model, object_info,
            Vec3f(float(to_mm(p.x())), float(to_mm(p.y())), float(z_height)), uv))
        {
          auto [sx, sy] = uv_to_svg(uv[0], uv[1]);
          svg << sx << "," << sy << " ";
          any = true;
        }
      }
      svg << "\" style=\"fill:none;stroke:blue;stroke-width:2\" />\n";
    }
    svg << "</svg>\n";
    svg.close();
  }

#if FILL_CHECKERED_DEBUG
  // Draw black-cell segments onto the UV texture and write debug PNG (verify segments lie on black cells).
  static void write_segments_to_texture(const Polylines &segments, const Model &model, const ObjInfo &object_info,
                                        coordf_t z_height, const std::string &obj_file_path, const std::string &output_path)
  {
    if (object_info.uv_map_pngs.empty())
      return;
    std::string tex_name = object_info.uv_map_pngs.begin()->second;
    boost::filesystem::path obj_path(obj_file_path);
    boost::filesystem::path tex_path = obj_path.parent_path() / tex_name;
    if (!boost::filesystem::exists(tex_path))
      return;
    std::ifstream ifs(tex_path.string(), std::ios::binary);
    if (!ifs)
      return;
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> file_buf(sz);
    if (!ifs.read(reinterpret_cast<char*>(file_buf.data()), sz))
      return;
    png::ReadBuf rbuf{file_buf.data(), file_buf.size()};
    png::ImageColorscale img;
    if (!png::decode_colored_png(rbuf, img) || img.rows == 0 || img.cols == 0)
      return;
    const size_t rowbytes = img.cols * img.bytes_per_pixel;
    auto set_pixel = [&](int px, int py, uint8_t r, uint8_t g, uint8_t b) {
      if (px < 0 || px >= (int)img.cols || py < 0 || py >= (int)img.rows) return;
      size_t idx = (size_t)py * rowbytes + (size_t)px * img.bytes_per_pixel;
      img.buf[idx] = r;
      img.buf[idx + 1] = g;
      if (img.bytes_per_pixel >= 3) img.buf[idx + 2] = b;
    };
    Eigen::Vector2f uv;
    for (const auto &pl : segments)
      for (const auto &p : pl.points)
        if (get_uv_for_point(model, object_info,
            Vec3f(float(to_mm(p.x())), float(to_mm(p.y())), float(z_height)), uv))
        {
          int px = (int)(uv[0] * (img.cols - 1) + 0.5f);
          int py = (int)((1.f - uv[1]) * (img.rows - 1) + 0.5f);
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              set_pixel(px + dx, py + dy, 255, 0, 0);
        }
    std::vector<uint8_t> rgb(img.rows * img.cols * 3);
    for (size_t y = 0; y < img.rows; ++y)
      for (size_t x = 0; x < img.cols; ++x)
      {
        size_t src = (y * img.cols + x) * img.bytes_per_pixel;
        size_t dst = (y * img.cols + x) * 3;
        rgb[dst] = img.buf[src];
        rgb[dst + 1] = img.buf[src + 1];
        rgb[dst + 2] = img.buf[src + 2];
      }
    png::write_rgb_to_file(output_path, img.cols, img.rows, rgb);
  }
#endif

  void dumpPolygon(const Slic3r::Polygon &poly, const std::string &label = "")
  {
    size_t n = poly.points.size();

    if (n > 1000000)
    {
      std::cerr << "[WARN] " << label << " polygon looks corrupted (" << n << " points)\n";
      return;
    }

    std::cout << "[PolygonMM] " << label << " has " << n << " points: ";
    for (size_t i = 0; i < n; i++)
    {
      const auto &p = poly.points[i];
      std::cout << "(" << to_mm(p.x()) << "mm," << to_mm(p.y()) << "mm) ";
    }
    std::cout << "\n";
  }

  static void print_polygon(const Slic3r::Polygon &poly)
  {
    std::cout << "Polygon with " << poly.points.size() << " points:\n";
    for (const auto &p : poly.points)
    {
      std::cout << "(" << p.x() << ", " << p.y() << ")\n";
    }
  }

  static void print_polyline(const Slic3r::Polyline &polyline)
  {
    printf("========print polyline=========\n");
    for (const auto &point : polyline.points)
    {
      std::cout << "(" << to_mm(point.x()) << "mm," << to_mm(point.y()) << "mm) \n";
    }
    printf("========print polyline end=========\n");
  }

  double dist(const Point &a, const Point &b)
  {
    double dx = b.x() - a.x();
    double dy = b.y() - a.y();
    return std::sqrt(dx * dx + dy * dy);
  }

  static Polyline traverse_along_contour(
      const Polygon &contour, const Point &start_point, double distance)
  {
    std::cout << "Traversing along contour from " << to_mm(start_point.x()) << "mm,"
              << to_mm(start_point.y()) << "mm for " << to_mm(distance) << "mm" << std::endl;
    Polyline result;
    const auto &pts = contour.points;
    size_t n = pts.size();

    if (contour.size() < 2)
      return result;

    result.points.push_back(start_point);

    // --- Find start segment ---
    size_t seg_index = 0;
    double t0 = 0.0;
    for (size_t i = 0; i < contour.size(); i++)
    {
      const Point &p1 = contour[i];
      const Point &p2 = contour[(i + 1) % contour.size()];
      double seg_len = dist(p1, p2);

      double d1 = dist(p1, start_point);
      double d2 = dist(start_point, p2);

      if (std::fabs((d1 + d2) - seg_len) < 1e-6)
      {
        seg_index = i;
        t0 = d1 / seg_len;
        break;
      }
    }

    // --- Start walking ---
    size_t i = seg_index;
    const Point *p1 = &contour[i];
    const Point *p2 = &contour[(i + 1) % contour.size()];
    double seg_len = dist(*p1, *p2);

    double remaining_in_seg = (1.0 - t0) * seg_len;
    double walked = 0.0;

    if (distance <= remaining_in_seg)
    {
      // final point lies in this segment
      double t = t0 + distance / seg_len;
      result.points.push_back(lerp(*p1, *p2, t));
      return result;
    }

    // consume the remainder of this segment
    walked += remaining_in_seg;
    result.points.push_back(*p2);
    i = (i + 1) % contour.size();

    while (walked < distance)
    {
      const Point &a = contour[i];
      const Point &b = contour[(i + 1) % contour.size()];
      double len = dist(a, b);

      if (walked + len >= distance)
      {
        // final point inside this segment
        double t = (distance - walked) / len;
        result.points.push_back(lerp(a, b, t));
        break;
      }

      walked += len;
      result.points.push_back(b);
      i = (i + 1) % contour.size();
    }

    return result;
  }

  Point nextPointOnCircle(Polygon &contour, const Point &start, double distance, double offset, bool clockwise = true)
  {
    auto mm = scale_(1);

    BoundingBox bb = contour.bounding_box();
    double cx = bb.center().x();
    double cy = bb.center().y();
    double r = (bb.size().x() + offset) / 2;

    // Current angle
    double theta = std::atan2(start.y() - cy, start.x() - cx);

    // Arc length -> angle offset
    double deltaTheta = distance / r;

    if (clockwise)
    {
      theta -= deltaTheta;
    }
    else
    {
      theta += deltaTheta;
    }

    // New coordinates
    double xNext = cx + r * std::cos(theta);
    double yNext = cy + r * std::sin(theta);

    return {xNext, yNext};
  }

  static Point moveAlongRadius(const Point &center, const Point &point, double newRadius)
  {
    // Direction vector from center to point
    double dx = point.x() - center.x();
    double dy = point.y() - center.y();

    // Current distance (length of radius)
    double len = std::sqrt(dx * dx + dy * dy);

    // Normalize direction
    double ux = dx / len;
    double uy = dy / len;

    // New point at desired radius
    Point moved;
    moved.x() = center.x() + ux * newRadius;
    moved.y() = center.y() + uy * newRadius;

    return moved;
  }

  static Polylines generate_segments_for_contour(Polygon &contour, int num_segments, double offset, coordf_t z_height)
  {
    Polylines segments;

    auto mm = scale_(1);
    int count_in_grid = int(z_height) % 18;
    bool shift = count_in_grid > 9;

    BoundingBox bb = contour.bounding_box();
    auto radius = double(bb.size().x() + offset) / 2;
    auto width = 2 * PI * radius / num_segments;

    Point previous_point;
    bool has_previous = false;

    // Generate segments along the contour
    for (int i = 0; i < num_segments; ++i)
    {
      Point start_point;
      if (!has_previous)
      {
        start_point = contour.points.front();
        if (shift)
        {
          start_point = nextPointOnCircle(contour, start_point, width, offset, false);
        }
        // std::cout << "Start Point(" << to_mm(start_point.x()) << ", " <<
        // to_mm(start_point.y()) << ")\n";
      }
      else
      {
        start_point = nextPointOnCircle(contour, previous_point, width, offset, false);
      }

      Point end = nextPointOnCircle(contour, start_point, width, offset, false);

      Polyline segment;
      segment.points.push_back(start_point);
      segment.points.push_back(end);

      segments.push_back(segment);

      previous_point = end;
      has_previous = true;
    }

    return segments;
  }

  Polyline joinSegments(const Polylines &outerSegments, const Polylines &innerSegments)
  {
    Polyline points;
    // points.reserve(outerSegments.size() * 4); // each outer segment contributes 4 points

    for (size_t i = 0; i < outerSegments.size() / 2; i++)
    {
      const Polyline &outer = outerSegments[i];

      // compute indices for inner segments
      int prevIndex = (i == 0) ? (int)innerSegments.size() - 1 : (int)i - 1;
      prevIndex = std::min(prevIndex, (int)innerSegments.size() - 1);

      int nextIndex = (i >= innerSegments.size()) ? 0 : (int)i;

      const Polyline &previousInner = innerSegments[prevIndex];
      const Polyline &nextInner = innerSegments[nextIndex];

      // push in the same order as TS code
      points.points.push_back(previousInner.points.back());
      points.points.push_back(outer.points.front());
      points.points.push_back(outer.points.back());
      points.points.push_back(nextInner.points.front());
    }

    return points;
  }

  Polyline joinSegmentsReverse(const Polylines &outerSegments, const Polylines &innerSegments)
  {
    Polyline points;
    // points.reserve(outerSegments.size() * 4); // each outer segment contributes 4 points

    for (size_t i = 0; i < (outerSegments.size() / 2) + 1; i++)
    {
      const Polyline &outer = outerSegments[i];

      // compute indices for inner segments
      int prevIndex = (i == 0) ? (int)innerSegments.size() - 1 : (int)i - 1;
      prevIndex = std::min(prevIndex, (int)innerSegments.size() - 1);

      int nextIndex = (i >= innerSegments.size()) ? 0 : (int)i;

      const Polyline &previousInner = innerSegments[i];
      const Polyline &nextInner = (i + 1 > innerSegments.size() - 1) ? innerSegments[0] : innerSegments[i + 1];

      // push in the same order as TS code
      points.points.push_back(previousInner.points.back());
      points.points.push_back(outer.points.front());
      points.points.push_back(outer.points.back());
      points.points.push_back(nextInner.points.front());
    }

    return points;
  }

  // Compute barycentric coordinates of P relative to triangle v0,v1,v2
  Eigen::Vector3f compute_barycentric(const Vec3f &P, const Vec3f &v0, const Vec3f &v1, const Vec3f &v2)
  {
    // printf("Computing barycentric for P(%f,%f,%f) in triangle V0(%f,%f,%f), V1(%f,%f,%f), V2(%f,%f,%f)\n",
    //        P[0], P[1], P[2],
    //        v0[0], v0[1], v0[2],
    //        v1[0], v1[1], v1[2],
    //        v2[0], v2[1], v2[2]);

    Eigen::Vector3f u = (v1 - v0);
    Eigen::Vector3f v = (v2 - v0);
    Eigen::Vector3f w = (P - v0);

    float uu = u.dot(u);
    float uv = u.dot(v);
    float vv = v.dot(v);
    float wu = w.dot(u);
    float wv = w.dot(v);

    float denom = uv * uv - uu * vv;
    if (denom == 0)
      return Eigen::Vector3f(-1, -1, -1); // degenerate triangle

    float alpha = (uv * wv - vv * wu) / denom;
    float beta = (uv * wu - uu * wv) / denom;
    float gamma = 1.0f - alpha - beta;

    return Eigen::Vector3f(alpha, beta, gamma);
  }

  bool get_uv_for_point(const Model &model, ObjInfo object_info, const Vec3f &P, Eigen::Vector2f &out_uv)
  {
    float best_dist2 = std::numeric_limits<float>::max();
    bool found = false;

    //std::cout << "Model has " << object_info.uvs.size() << " uvs\n";

    if(object_info.uvs.size() == 0)
    {
      std::cout << "No UVs available in object_info\n";
      return false;
    }

    size_t global_face_idx = 0;
    for (size_t i = 0; i < model.objects.size(); ++i)
    {
      for (size_t j = 0; j < model.objects[i]->volumes.size(); ++j)
      {
        const auto &mesh = model.objects[i]->volumes[j]->mesh();

        for (size_t t = 0; t < mesh.its.indices.size(); ++t)
        {
          if (global_face_idx >= object_info.uvs.size())
            break;
          const auto &tri = mesh.its.indices[t]; // typically Eigen::Vector3i

          // Vertices of triangle
          Eigen::Vector3f v0 = mesh.its.vertices[tri[0]].cast<float>();
          Eigen::Vector3f v1 = mesh.its.vertices[tri[1]].cast<float>();
          Eigen::Vector3f v2 = mesh.its.vertices[tri[2]].cast<float>();

          const auto &uv0 = object_info.uvs[global_face_idx][0];
          const auto &uv1 = object_info.uvs[global_face_idx][1];
          const auto &uv2 = object_info.uvs[global_face_idx][2];

          Eigen::Vector3f bary = compute_barycentric(P, v0, v1, v2);
          //printf("Barycentric coords: (%f, %f, %f)\n", bary[0], bary[1], bary[2]);

          // Check if point is inside triangle (allow small tolerance)
          if (bary[0] >= -1e-4 && bary[1] >= -1e-4 && bary[2] >= -1e-4)
          {
            //printf("Point is inside the triangle\n");
            out_uv = bary[0] * uv0 + bary[1] * uv1 + bary[2] * uv2;
            return true;
          }
          else
          {
            //printf("Point is outside the triangle, checking closest point\n");
            // Optionally: find nearest triangle if point not exactly inside
            Eigen::Vector3f closest_point = bary[0] * v0 + bary[1] * v1 + bary[2] * v2;
            //printf("Closest point on triangle: (%f, %f, %f)\n", closest_point[0], closest_point[1], closest_point[2]);
            //printf("Original point P: (%f, %f, %f)\n", P[0], P[1], P[2]);
            float dist2 = (closest_point - P).squaredNorm();
            //printf("Distance squared to triangle: %f\n", dist2);
            if (dist2 < best_dist2)
            {
              //printf("Found a closer triangle with dist2 = %f (previous best was %f)\n", dist2, best_dist2);
              best_dist2 = dist2;
              out_uv = bary[0] * uv0 + bary[1] * uv1 + bary[2] * uv2;
              //printf("Updating closest UV to (%f, %f) with dist2 = %f\n", out_uv[0], out_uv[1], dist2);

              found = true;
            }
          }
          ++global_face_idx;
        }
      }
    }

    return found;
  }

  inline Eigen::Vector2f next_grid_boundary(const Eigen::Vector2f &uv, float cell_size)
  {
    Eigen::Vector2f result;
    // Next U boundary
    result[0] = (std::floor(uv[0] / cell_size) + 1) * cell_size;
    // Next V boundary
    result[1] = (std::floor(uv[1] / cell_size) + 1) * cell_size;
    return result;
  }

  // Return grid cell indices (i, j) for a UV point. Checkerboard: black if (i+j)%2==0.
  inline std::pair<int, int> uv_to_grid_cell(const Eigen::Vector2f &uv, float cell_size)
  {
    int gi = static_cast<int>(std::floor(uv[0] / cell_size));
    int gj = static_cast<int>(std::floor(uv[1] / cell_size));
    return {gi, gj};
  }

  inline bool is_black_cell(int grid_i, int grid_j)
  {
    return (grid_i + grid_j) % 2 == 0;
  }

  static constexpr float GRID_CELL_SIZE = 1.0f / 64.0f;

  struct ContourPointInfo {
    Point xy;
    Eigen::Vector2f uv;
    int grid_i{0}, grid_j{0};
    bool is_black{false};
    bool has_uv{false};
  };

  static Polylines generate_segments_for_contour(Polygon &contour, Model &model, ObjInfo object_info, double offset, coordf_t z_height)
  {
    Polylines segments;
    const auto &points = contour.points;
    if (points.empty() || object_info.uvs.empty())
      return segments;

    const float cell_size = GRID_CELL_SIZE;
    std::vector<ContourPointInfo> infos;
    infos.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i)
    {
      const Point &point = points[i];
      ContourPointInfo info;
      info.xy = point;
      info.has_uv = get_uv_for_point(model, object_info,
        Vec3f(float(to_mm(point.x())), float(to_mm(point.y())), float(z_height)), info.uv);
      if (info.has_uv)
      {
        auto cell = uv_to_grid_cell(info.uv, cell_size);
        info.grid_i = cell.first;
        info.grid_j = cell.second;
        info.is_black = is_black_cell(info.grid_i, info.grid_j);
      }
      else
        info.is_black = false;
      infos.push_back(info);
    }

    Polyline current;
    size_t first_black_segment_idx = size_t(-1); // index in segments of the first segment (for wrap merge)

    for (size_t i = 0; i < points.size(); ++i)
    {
      size_t next_i = (i + 1) % points.size();
      bool cur_black = infos[i].is_black;
      bool next_black = infos[next_i].is_black;

      if (cur_black && next_black)
      {
        if (current.points.empty())
        {
          if (first_black_segment_idx == size_t(-1))
            first_black_segment_idx = segments.size();
          current.points.push_back(infos[i].xy);
        }
        current.points.push_back(infos[next_i].xy);
      }
      else if (cur_black && !next_black)
      {
        if (!current.points.empty())
        {
          segments.push_back(std::move(current));
          current = Polyline();
        }
      }
      else if (!cur_black && next_black)
      {
        current.points.push_back(infos[next_i].xy);
      }
    }

    if (!current.points.empty())
    {
      if (first_black_segment_idx != size_t(-1) && infos[0].is_black)
      {
        Polyline &first = segments[first_black_segment_idx];
        first.points.insert(first.points.begin(), current.points.begin(), current.points.end() - 1);
      }
      else
        segments.push_back(std::move(current));
    }

    return segments;
  }

  static Polylines generate_infill_layers(Polygon &contour, Polygon &hole, coordf_t z_height, Model &model, ObjInfo object_info)
  {
    auto offset = 0; // 0.2 * scale_(1); // 0.6mm inward offset

    if ((int(z_height) - 1) % 3 == 0)
    {
      offset = 0;
    }

    Polylines outer_segments = generate_segments_for_contour(contour, model, object_info, -offset, z_height);
    return outer_segments;
  }

  void print_model_details(const Slic3r::Model &model)
  {
    std::cout << "Model has " << model.objects.size() << " objects\n";

    for (size_t i = 0; i < model.objects.size(); ++i)
    {
      const auto *obj = model.objects[i];
      std::cout << " Object " << i << " has " << obj->volumes.size() << " volumes\n";

      for (size_t j = 0; j < obj->volumes.size(); ++j)
      {
        const auto *vol = obj->volumes[j];
        std::cout << "  Volume " << j << " name: " << vol->name << "\n";

        const auto &mesh = vol->mesh();
        std::cout << "   Vertices: " << mesh.its.vertices.size() << "\n";

        // Print first few vertices
        for (size_t k = 0; k < std::min<size_t>(mesh.its.vertices.size(), 5); ++k)
        {
          const auto &v = mesh.its.vertices[k];
          std::cout << "    v" << k << " = ("
                    << v(0) << ", " << v(1) << ", " << v(2) << ")\n";
        }
      }
    }
  }

  void FillCheckered::_fill_surface_single(
      const FillParams &params,
      unsigned int thickness_layers,
      const std::pair<float, Point> &direction,
      ExPolygon expolygon,
      Polylines &polylines_out)
  {
    auto mm = scale_(1);

    const char *obj_file_path = params.sparse_infill_checkered_file;
    //std::cout << "Found file path for checkered infill: " << obj_file_path << "\n";

    // check if file path is valid or empty, return if invalid
    if (obj_file_path == nullptr || strlen(obj_file_path) == 0)
    {
      std::cout << "Invalid or empty obj file path for checkered infill. Skipping infill.\n";
      return;
    }

    Slic3r::Model model;
    Slic3r::ObjInfo objinfo; // holds vertex color info if present
    std::string message;

    Slic3r::load_obj(obj_file_path, &model, objinfo, message);

    //Try dumping UVs if present
    // if (!objinfo.uvs.empty()) {
    //     std::cout << "UV count = " << objinfo.uvs.size() << "\n";
    //     for (size_t i = 0; i < std::min<size_t>(5, objinfo.uvs.size()); ++i) {
    //         auto uv = objinfo.uvs[i];
    //         std::cout << "  uv" << i << ": (" << uv[0] << ", " << uv[1] << ")\n";
    //     }
    // } else {
    //     std::cout << "No UVs found in ObjInfo\n";
    //     return;
    // }

    // print_model_details(model);

    Polygon outer_contour = expolygon.contour;
    Polygon hole;

    Polylines polylines = generate_infill_layers(outer_contour, hole, this->z, model, objinfo);

#if FILL_CHECKERED_DEBUG
    if (!polylines.empty())
    {
      std::string base_path = obj_file_path;
      size_t slash = base_path.find_last_of("/\\");
      std::string dir = (slash != std::string::npos) ? base_path.substr(0, slash + 1) : "";
      std::string xy_svg = dir + "debug_checkered_xy.svg";
      std::string uv_svg = dir + "debug_checkered_uv.svg";
      std::string tex_png = dir + "debug_checkered_texture.png";
      write_polylines_to_svg(polylines, xy_svg, &outer_contour);
      write_uv_polylines_to_svg(polylines, model, objinfo, this->z, uv_svg);
      write_segments_to_texture(polylines, model, objinfo, this->z, obj_file_path, tex_png);
    }
#endif

    append(polylines_out, std::move(polylines));
  }
} // namespace Slic3r