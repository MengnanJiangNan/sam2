#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import sys
import glob
from tqdm import tqdm
import multiprocessing
from functools import partial
from PIL import Image
import trimesh # Need trimesh for robust GLB loading

# Ensure script directory is in the path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Import utility functions
from utils import sphere_hammersley_sequence
# Assuming generate_coco_annotations might need adaptation or removal if not applicable
# from generate_coco_annotations import create_coco_dataset

# Color palette for instance masks
INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 165, 0),
    (255, 215, 0), (173, 255, 47), (0, 250, 154), (70, 130, 180), (218, 112, 214)
]

def setup_object_output_dirs(base_dir, object_id):
    """
    Sets up the output directory structure for a single object.
    """
    # Normalize base path first
    base_dir_norm = os.path.normpath(base_dir)
    object_dir = os.path.join(base_dir_norm, object_id)
    camera_dir = os.path.join(object_dir, 'camera')
    rgb_dir = os.path.join(object_dir, 'rgb')
    normal_dir = os.path.join(object_dir, 'normal_map')
    # ccm_dir = os.path.join(object_dir, 'ccm') # CCM not directly generated here
    mask_dir = os.path.join(object_dir, 'mask')
    instance_mask_dir = os.path.join(object_dir, 'instance_mask')
    depth_dir = os.path.join(object_dir, 'depth')

    # Create all required directories
    for dir_path in [object_dir, camera_dir, rgb_dir, normal_dir, mask_dir, instance_mask_dir, depth_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # return object_dir, camera_dir, rgb_dir, normal_dir, ccm_dir, mask_dir, instance_mask_dir, depth_dir
    return object_dir, camera_dir, rgb_dir, normal_dir, mask_dir, instance_mask_dir, depth_dir


def save_image(path, image_pil):
    """Save PIL image."""
    image_pil.save(path)

def save_depth(path, depth_image_o3d):
    """Save depth map (float32) as PNG, converting to uint16 for better range."""
    depth_np = np.asarray(depth_image_o3d)
    # Scale to uint16 range, handling potential infinite depth
    valid_depth = depth_np[np.isfinite(depth_np)]
    if valid_depth.size > 0:
        min_d, max_d = valid_depth.min(), valid_depth.max()
        if max_d > min_d:
             # Normalize finite depth to [0, 1] and scale to uint16
            depth_norm = (depth_np - min_d) / (max_d - min_d)
            depth_norm[~np.isfinite(depth_np)] = 1.0 # Map inf to max value
            depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        else: # Handle case where depth is constant
             depth_uint16 = np.zeros_like(depth_np, dtype=np.uint16)
    else: # Handle case where there's no valid depth
        depth_uint16 = np.zeros_like(depth_np, dtype=np.uint16)

    Image.fromarray(depth_uint16).save(path)

def save_mask(path, depth_image_o3d):
    """Generate and save a binary mask from depth."""
    depth_np = np.asarray(depth_image_o3d)
    mask = (depth_np > 0).astype(np.uint8) * 255
    mask_image_pil = Image.fromarray(mask, mode='L')
    mask_image_pil.save(path)

def get_camera_matrices(fov_deg, eye, center, up, width, height):
    """Calculate projection and view matrices."""
    # Open3D uses vertical FOV
    fovy = fov_deg * (np.pi / 180.0)
    aspect = width / height
    near = 0.1
    far = 100.0 # Adjust far plane as needed

    # Projection Matrix (Perspective)
    f = 1.0 / np.tan(fovy / 2.0)
    proj = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float64)


    # View Matrix (LookAt)
    f = (center - eye)
    f = f / np.linalg.norm(f)
    u = np.array(up, dtype=np.float64)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    view = np.array([
        [s[0], s[1], s[2], -np.dot(s, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # World-to-Camera (View Matrix)
    world_to_cam = view

    return proj, world_to_cam

def render_object_open3d(mesh_path, num_views, output_dirs, resolution=512, start_id=0, end_id=None):
    """
    Render an object using Open3D from multiple viewpoints, using Open3D's model loader.
    """
    # Unpack output directories
    object_dir, camera_dir, rgb_dir, normal_dir, mask_dir, instance_mask_dir, depth_dir = output_dirs
    width, height = resolution, resolution

    # Initialize Renderer
    render = rendering.OffscreenRenderer(width, height)
    o3d_meshes_render = [] # Store meshes used for rendering passes
    o3d_meshes_bbox = [] # Store meshes just for bounding box calculation

    try:
        # --- Use Open3D's Model Loader ---
        try:
            # read_triangle_model can load textures and materials
            model = o3d.io.read_triangle_model(mesh_path, True) # True enables material/texture loading
            if not model.meshes:
                 print(f"Warning: o3d.io.read_triangle_model loaded no meshes from {mesh_path}.")
                 # Try legacy loader as fallback? Or trimesh? For now, let's return False.
                 return False

            # Keep original loaded meshes separate for bounding box
            for mesh_info in model.meshes:
                 mesh_o3d_orig = mesh_info.mesh
                 if mesh_o3d_orig.has_vertices() and mesh_o3d_orig.has_triangles():
                     o3d_meshes_bbox.append(mesh_o3d_orig)

            if not o3d_meshes_bbox:
                 print(f"Error: No valid meshes found for bounding box in {mesh_path}")
                 return False

        except Exception as e:
            print(f"Error: Failed to load {mesh_path} with o3d.io.read_triangle_model: {e}")
            # Could add fallback to trimesh here if needed
            return False

        # --- Scene Setup ---
        render.scene.set_background([0.0, 0.0, 0.0, 1.0]) # Black background
        # Increase sun light intensity
        render.scene.scene.set_sun_light([0.7, 0.7, 0.7], [0.5, -0.5, -0.5], 120000) # Increased intensity
        render.scene.scene.enable_sun_light(True)
        # Use a lighting profile which might include ambient contribution
        render.scene.set_lighting(rendering.Scene.LightingProfile.MED_SHADOWS, (0, 0, 0)) # Try MED_SHADOWS

        # Define default materials for fallbacks if needed, although model.materials should be used
        material_default = rendering.MaterialRecord()
        material_default.shader = "defaultLit"
        material_default.base_color = [0.9, 0.9, 0.9, 1.0] # Slightly brighter default grey

        material_normal = rendering.MaterialRecord()
        material_normal.shader = "defaultUnlit"
        material_normal.base_color = [1.0, 1.0, 1.0, 1.0]

        material_instance = rendering.MaterialRecord()
        material_instance.shader = "defaultUnlit"
        material_instance.base_color = [1.0, 1.0, 1.0, 1.0]

        # --- Process Loaded Model Meshes ---
        all_min_bounds = []
        all_max_bounds = []
        rgb_geom_names = []
        normal_geom_names = []
        instance_geom_names = []

        for i, mesh_info in enumerate(model.meshes):
            mesh_o3d = mesh_info.mesh # The mesh loaded by Open3D
            material_idx = mesh_info.material_idx

            if not mesh_o3d.has_vertices() or not mesh_o3d.has_triangles():
                print(f"  Skipping mesh {i} due to missing vertices or triangles.")
                continue

            # Ensure normals are computed
            mesh_o3d.compute_vertex_normals()

            # Get bounding box
            bounds = mesh_o3d.get_axis_aligned_bounding_box()
            all_min_bounds.append(bounds.min_bound)
            all_max_bounds.append(bounds.max_bound)

            # Get the material assigned by the loader
            if material_idx < len(model.materials):
                rgb_material = model.materials[material_idx]
                # Ensure shader is set (sometimes defaults might be missing)
                if not rgb_material.shader:
                     rgb_material.shader = "defaultLit"
                # Check if texture was loaded
                # print(f"  Mesh {i} Material: {rgb_material}")
                # if rgb_material.albedo_img is not None:
                #      print(f"    -> Has albedo texture.")

            else:
                print(f"  Warning: Material index {material_idx} out of bounds for mesh {i}. Using default material.")
                rgb_material = material_default


            # -- Add geometries to scene for different passes --
            # We need distinct geometry objects if we modify vertex colors

            # 1. RGB Pass: Use the loaded mesh and its assigned material
            rgb_name = f"rgb_mesh_{i}"
            rgb_geom_names.append(rgb_name)
            render.scene.add_geometry(rgb_name, mesh_o3d, rgb_material)
            o3d_meshes_render.append(mesh_o3d) # Add to list of rendered meshes

            # 2. Normal Pass: Create a copy and assign normal colors
            normal_name = f"normal_mesh_{i}"
            normal_geom_names.append(normal_name)
            mesh_normal = o3d.geometry.TriangleMesh(mesh_o3d) # Create copy
            normals = np.asarray(mesh_normal.vertex_normals)
            normal_colors = (normals + 1.0) * 0.5 # Map normals [-1, 1] to colors [0, 1]
            mesh_normal.vertex_colors = o3d.utility.Vector3dVector(normal_colors)
            render.scene.add_geometry(normal_name, mesh_normal, material_normal) # Use specific normal material

            # 3. Instance Pass: Create a copy and assign instance colors
            instance_name = f"instance_mesh_{i}"
            instance_geom_names.append(instance_name)
            mesh_instance = o3d.geometry.TriangleMesh(mesh_o3d) # Create copy
            instance_color_rgb = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
            instance_color = [c / 255.0 for c in instance_color_rgb] # Normalize to [0, 1]
            mesh_instance.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(instance_color, (len(mesh_instance.vertices), 1))
            )
            render.scene.add_geometry(instance_name, mesh_instance, material_instance) # Use specific instance material


        if not o3d_meshes_render:
             print(f"Error: No valid meshes could be prepared for rendering from {mesh_path}")
             return False

        # Create the combined bounding box manually from original bbox meshes
        global_min = np.min(all_min_bounds, axis=0)
        global_max = np.max(all_max_bounds, axis=0)
        combined_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=global_min, max_bound=global_max)

        scene_center = combined_bounds.get_center()
        scene_extent = combined_bounds.get_extent()
        max_extent = np.max(scene_extent) if scene_extent.any() else 1.0


        # --- Camera and View Setup --- (Keep existing logic) ---
        fov_deg = 60.0 # Field of view in degrees
        radius = max_extent * 1.5 # Heuristic for camera distance
        if end_id is None: end_id = start_id + num_views - 1
        actual_num_views = end_id - start_id + 1
        views = []
        offset = (np.random.rand(), np.random.rand())
        for i in range(actual_num_views):
            yaw_norm, pitch_norm = sphere_hammersley_sequence(i, actual_num_views, offset)
            yaw = yaw_norm * 2 * np.pi
            pitch = (pitch_norm - 0.5) * np.pi
            eye_x = scene_center[0] + radius * np.cos(pitch) * np.sin(yaw)
            eye_y = scene_center[1] + radius * np.sin(pitch)
            eye_z = scene_center[2] + radius * np.cos(pitch) * np.cos(yaw)
            eye = np.array([eye_x, eye_y, eye_z])
            up = np.array([0.0, 1.0, 0.0])
            if np.linalg.norm(eye - scene_center) < 1e-6: eye[1] += 0.1
            proj_matrix, view_matrix = get_camera_matrices(fov_deg, eye, scene_center, up, width, height)
            transform_matrix = view_matrix.tolist()
            view_id = start_id + i
            views.append({ "file_name": f"{view_id:03d}.png", "yaw": yaw, "pitch": pitch, "radius": radius, "eye": eye.tolist(), "center": scene_center.tolist(), "up": up.tolist(), "transform_matrix": transform_matrix })
        # --- Save camera parameters (Keep existing logic) ---
        transforms_data = { "camera_angle_x": fov_deg * (np.pi / 180.0), "frames": [ { "file_path": os.path.join('rgb', v["file_name"]), "transform_matrix": v["transform_matrix"] } for v in views ] }
        with open(os.path.join(camera_dir, "transforms.json"), "w") as f: json.dump(transforms_data, f, indent=2)

        # --- Rendering Loop (Adjust geometry visibility control) ---
        for view_info in tqdm(views, desc=f"  Rendering views for {os.path.basename(mesh_path)}", leave=False):
            eye = np.array(view_info["eye"])
            center = np.array(view_info["center"])
            up = np.array(view_info["up"])
            render.setup_camera(fov_deg, center, eye, up)

            # Render RGB: Show only RGB geometries
            for name in normal_geom_names + instance_geom_names: render.scene.show_geometry(name, False)
            for name in rgb_geom_names: render.scene.show_geometry(name, True)
            rgb_o3d = render.render_to_image()
            save_image(os.path.join(rgb_dir, view_info["file_name"]), Image.fromarray(np.asarray(rgb_o3d)))

            # Render Depth (can use RGB setup)
            depth_o3d = render.render_to_depth_image(z_in_view_space=False)
            save_depth(os.path.join(depth_dir, view_info["file_name"]), depth_o3d)
            save_mask(os.path.join(mask_dir, view_info["file_name"]), depth_o3d) # Mask from depth

            # Render Normal Map: Show only Normal geometries
            for name in rgb_geom_names + instance_geom_names: render.scene.show_geometry(name, False)
            for name in normal_geom_names: render.scene.show_geometry(name, True)
            normal_o3d = render.render_to_image()
            save_image(os.path.join(normal_dir, view_info["file_name"]), Image.fromarray(np.asarray(normal_o3d)))

            # Render Instance Map: Show only Instance geometries
            if len(o3d_meshes_render) > 1: # Only if multiple parts were actually added
                 for name in rgb_geom_names + normal_geom_names: render.scene.show_geometry(name, False)
                 for name in instance_geom_names: render.scene.show_geometry(name, True)
                 instance_o3d = render.render_to_image()
                 save_image(os.path.join(instance_mask_dir, view_info["file_name"]), Image.fromarray(np.asarray(instance_o3d)))
            else: # Fallback to mask if only one part
                 save_mask(os.path.join(instance_mask_dir, view_info["file_name"]), depth_o3d)

    except Exception as e:
        print(f"Error rendering {mesh_path}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Explicitly clear scene geometries to release resources, especially in loops
        render.scene.clear_geometry()
        # Deleting the renderer might also help, though Python's GC should handle it
        del render

    return True

def batch_render(mesh_dir, num_views, output_dir, num_test_models=None,
                 start_index=None, end_index=None, gpu_id=None, num_processes=1,
                 resolution=512, start_id=0, end_id=None):
    """
    Batch render multiple objects using Open3D.
    NOTE: Parallel processing (num_processes > 1) is currently disabled due to potential instability.
    """
    if num_processes > 1:
        print("Warning: Parallel processing (--num_processes > 1) is currently disabled for Open3D rendering due to potential instability. Running sequentially.")
        num_processes = 1

    # Get list of all mesh files - support both .obj and .glb
    print(f"Searching for meshes in: {mesh_dir}")
    mesh_files = glob.glob(os.path.join(mesh_dir, "**/*.obj"), recursive=True) + \
                 glob.glob(os.path.join(mesh_dir, "**/*.glb"), recursive=True)
    mesh_files = sorted(list(set(mesh_files))) # Sort and remove duplicates
    print(f"Found {len(mesh_files)} mesh files.")

    if not mesh_files:
        print("No mesh files found. Exiting.")
        return

    # Apply start/end index for files if specified
    total_files = len(mesh_files)
    files_to_process = mesh_files
    if start_index is not None or end_index is not None:
        start_idx = start_index if start_index is not None else 0
        end_idx = end_index if end_index is not None else total_files - 1
        if start_idx < 0: start_idx = 0
        if end_idx >= total_files: end_idx = total_files - 1
        if start_idx > end_idx:
            print(f"Error: start_index ({start_idx}) > end_index ({end_idx}). No files to process.")
            return
        files_to_process = mesh_files[start_idx : end_idx + 1]
        print(f"Processing files from index {start_idx} to {end_idx} ({len(files_to_process)} files).")


    # Limit to test models if specified (overrides start/end index)
    if num_test_models is not None:
        if num_test_models < len(files_to_process):
             print(f"Limiting to {num_test_models} test models (random selection).")
             import random
             files_to_process = random.sample(files_to_process, num_test_models)
        else:
             print(f"Requested {num_test_models} test models, but only {len(files_to_process)} available. Processing all.")


    if not files_to_process:
         print("No files selected for processing.")
         return

    # Process each mesh sequentially
    success_count = 0
    failed_objects = []
    for mesh_path in tqdm(files_to_process, desc="Rendering objects"):
        object_id = os.path.splitext(os.path.basename(mesh_path))[0]
        # Use relative path from mesh_dir to create intermediate dirs if needed
        relative_path = os.path.relpath(os.path.dirname(mesh_path), mesh_dir)
        object_output_base = os.path.join(output_dir, relative_path)

        output_dirs = setup_object_output_dirs(object_output_base, object_id)

        try:
            success = render_object_open3d(mesh_path, num_views, output_dirs,
                                         resolution=resolution, start_id=start_id, end_id=end_id)
            if success:
                # print(f"Successfully rendered {object_id}")
                success_count += 1
            else:
                print(f"Failed to render {object_id}")
                failed_objects.append(object_id)
        except Exception as e:
            print(f"Critical error rendering {object_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_objects.append(object_id)

    print(f"\nBatch rendering completed.")
    print(f"Successfully rendered: {success_count}/{len(files_to_process)}")
    if failed_objects:
        print(f"Failed objects ({len(failed_objects)}): {', '.join(failed_objects)}")

    # Add COCO generation call here if adapted for Open3D output structure
    # print("\nAttempting to generate COCO annotations...")
    # create_coco_dataset(output_dir, os.path.join(output_dir, "coco_annotations.json"))

def main():
    parser = argparse.ArgumentParser(description="Batch render 3D meshes using Open3D")
    parser.add_argument("--mesh_dir", required=True, help="Directory containing .obj or .glb files (can be nested)")
    parser.add_argument("--output_dir", required=True, help="Output directory for rendered images")
    parser.add_argument("--num_views", type=int, default=24, help="Number of views to render per object if end_id is not set")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of rendered images (width and height)")
    parser.add_argument("--num_test_models", type=int, help="Number of models to render (randomly selected, overrides start/end index)")
    parser.add_argument("--start_index", type=int, help="Start index for mesh files list (0-based)")
    parser.add_argument("--end_index", type=int, help="End index for mesh files list (inclusive)")
    parser.add_argument("--gpu_id", type=int, help="GPU ID to use (Note: Open3D GPU selection might be automatic)")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes (Note: >1 currently disabled)")
    parser.add_argument("--start_id", type=int, default=0, help="Start view ID for numbering.")
    parser.add_argument("--end_id", type=int, default=None, help="End view ID for numbering (inclusive). If None, views = start_id + num_views - 1.")

    args = parser.parse_args()

    # Calculate actual number of views based on start/end_id or num_views
    if args.end_id is None:
        args.end_id = args.start_id + args.num_views - 1
    elif args.end_id < args.start_id:
        print("Error: end_id must be greater than or equal to start_id")
        exit(1)
    actual_num_views_per_object = args.end_id - args.start_id + 1
    print(f"Configuration:")
    print(f"  Mesh Source: {args.mesh_dir}")
    print(f"  Output Destination: {args.output_dir}")
    print(f"  Views per Object: {actual_num_views_per_object} (IDs {args.start_id} to {args.end_id})")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    if args.start_index is not None or args.end_index is not None:
         print(f"  File Index Range: {args.start_index if args.start_index is not None else 'start'} to {args.end_index if args.end_index is not None else 'end'}")
    if args.num_test_models is not None:
         print(f"  Test Mode: {args.num_test_models} models")


    batch_render(
        args.mesh_dir,
        actual_num_views_per_object, # Pass the calculated number of views
        args.output_dir,
        args.num_test_models,
        args.start_index,
        args.end_index,
        args.gpu_id,
        args.num_processes,
        args.resolution,
        start_id=args.start_id, # Pass start_id for numbering
        end_id=args.end_id      # Pass end_id for numbering range
    )

if __name__ == "__main__":
    main() 