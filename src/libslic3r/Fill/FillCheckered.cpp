#include "../ClipperUtils.hpp"
#include "../ShortestPath.hpp"
#include "../Surface.hpp"
#include <cmath>

#include "FillCheckered.hpp"
#include "libslic3r/Format/OBJ.hpp"
#include "libslic3r/Model.hpp"
#include <iostream>

// static const double PI = 3.14159265358979323846;

namespace Slic3r {
inline double to_mm(coord_t v) {
    auto mm = scale_(1);
    return double(v) / mm;
}

void dumpPolygon(const Slic3r::Polygon &poly, const std::string &label = "") {
    size_t n = poly.points.size();

    if (n > 1000000) {
        std::cerr << "[WARN] " << label << " polygon looks corrupted (" << n << " points)\n";
        return;
    }

    std::cout << "[PolygonMM] " << label << " has " << n << " points: ";
    for (size_t i = 0; i < n; i++) {
        const auto &p = poly.points[i];
        std::cout << "(" << to_mm(p.x()) << "mm," << to_mm(p.y()) << "mm) ";
    }
    std::cout << "\n";
}

static void print_polygon(const Slic3r::Polygon &poly) {
    std::cout << "Polygon with " << poly.points.size() << " points:\n";
    for (const auto &p : poly.points) {
        std::cout << "(" << p.x() << ", " << p.y() << ")\n";
    }
}

static void print_polyline(const Slic3r::Polyline &polyline) {
    printf("========print polyline=========\n");
    for (const auto &point : polyline.points) {
        std::cout << "(" << to_mm(point.x()) << "mm," << to_mm(point.y()) << "mm) \n";
    }
    printf("========print polyline end=========\n");
}


void print_model_details(const Slic3r::Model &model) {
    std::cout << "Model has " << model.objects.size() << " objects\n";

    for (size_t i = 0; i < model.objects.size(); ++i) {
        const auto* obj = model.objects[i];
        std::cout << " Object " << i << " has " << obj->volumes.size() << " volumes\n";

        for (size_t j = 0; j < obj->volumes.size(); ++j) {
            const auto* vol = obj->volumes[j];
            std::cout << "  Volume " << j << " name: " << vol->name << "\n";

            const auto& mesh = vol->mesh();
            std::cout << "   Vertices: " << mesh.its.vertices.size() << "\n";

            // Print first few vertices
            for (size_t k = 0; k < std::min<size_t>(mesh.its.vertices.size(), 5); ++k) {
                const auto& v = mesh.its.vertices[k];
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
    Polylines &polylines_out
) {    
    auto mm = scale_(1);

    const char* obj_file_path = params.sparse_infill_checkered_file;
    printf("Checkered obj file path: %s\n", obj_file_path);

    //check if file path is valid or empty, return if invalid
    if (obj_file_path == nullptr || strlen(obj_file_path) == 0) {
        std::cout << "Invalid or empty obj file path for checkered infill. Skipping infill.\n";
        return;
    }

    Slic3r::Model model;
    Slic3r::ObjInfo objinfo;       // holds vertex color info if present
    std::string message;

    Slic3r::load_obj(obj_file_path, &model, objinfo, message);

    std::cout << "Model has " << model.objects.size() << " objects\n";

    // Try dumping UVs if present
    if (!objinfo.uvs.empty()) {
        std::cout << "UV count = " << objinfo.uvs.size() << "\n";
        for (size_t i = 0; i < std::min<size_t>(5, objinfo.uvs.size()); ++i) {
            auto uv = objinfo.uvs[i];
            std::cout << "  uv" << i << ": (" << uv[0] << ", " << uv[1] << ")\n";
        }
    } else {
        std::cout << "No UVs found in ObjInfo\n";
    }

    //print_model_details(model);

    // get outer contour and scale inwards by one layer height
    // Polygon outer_contour = offset(expolygon.contour, -0.2 * mm)[0];
    // Polygon hole = offset(expolygon.holes[0], 0.2 * mm)[0];
    Polygon outer_contour = expolygon.contour;
    Polygon hole = expolygon.holes[0];
  }
} // namespace Slic3r