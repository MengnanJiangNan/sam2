#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from subprocess import DEVNULL, call
from utils import sphere_hammersley_sequence
import shutil  # Import for directory removal
import sys
import glob  # For finding all EXR files

# Ensure script directory is in the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the depth to CCM conversion function
try:
    from depth_to_ccm import batch_process_depth_to_ccm
except ImportError:
    print("Warning: Could not import depth_to_ccm module. CCM generation will be skipped.")
    batch_process_depth_to_ccm = None

'''
python mesh_render.py --mesh 3D/textured_mesh.glb --num_views 24
'''

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def get_camera_matrix(yaw, pitch, radius):
    """
    Calculate camera extrinsic matrix from spherical coordinates (yaw, pitch, radius).
    The extrinsic matrix (4×4 homogeneous transformation matrix) transforms world coordinates to camera coordinates.
    """
    # Calculate camera position on sphere
    x = radius * np.cos(pitch) * np.cos(yaw)
    y = radius * np.sin(pitch)
    z = radius * np.cos(pitch) * np.sin(yaw)
    cam_pos = np.array([x, y, z])

    # Camera always points to origin
    forward = -cam_pos / np.linalg.norm(cam_pos)
    # Define up direction (positive y-axis)
    up = np.array([0, 1, 0])
    # Calculate right direction
    right = np.cross(up, forward)
    # If forward and up are parallel (right is zero vector), redefine up
    if np.linalg.norm(right) == 0:
        up = np.array([0, 0, 1])
        right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Rotation matrix: use right, up, forward as basis vectors
    R = np.stack([right, up, forward], axis=1)
    T = cam_pos.reshape(3, 1)

    # Construct 4×4 extrinsic matrix: R and translation part -R^T * T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ T[:, 0]
    return extrinsic.tolist()

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def setup_output_dirs():
    """
    Sets up the output directory structure. Clears and recreates if it exists.
    Creates separate subdirectories for different types of outputs.
    
    Returns:
        Tuple containing paths to the base and subdirectories
    """
    base_dir = 'render_output'
    
    # Create specific output subdirectories
    camera_dir = os.path.join(base_dir, 'camera')
    rgb_dir = os.path.join(base_dir, 'rgb')
    depth_dir = os.path.join(base_dir, 'depth_map')
    normal_dir = os.path.join(base_dir, 'normal_map')
    ccm_dir = os.path.join(base_dir, 'ccm')
    face_id_dir = os.path.join(base_dir, 'face_id')
    face_id_mapping_dir = os.path.join(base_dir, 'face_id_mapping')
    
    # If base directory exists, remove it
    if os.path.exists(base_dir):
        print(f"Removing existing output directory: {base_dir}")
        shutil.rmtree(base_dir)
    
    # Create base directory and all subdirectories
    print(f"Creating output directory structure in: {base_dir}")
    os.makedirs(base_dir)
    os.makedirs(camera_dir)
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    os.makedirs(normal_dir)
    os.makedirs(ccm_dir)
    os.makedirs(face_id_dir)
    os.makedirs(face_id_mapping_dir)
    
    return base_dir, camera_dir, rgb_dir, depth_dir, normal_dir, ccm_dir, face_id_dir, face_id_mapping_dir

def visualize_maps(base_dir, depth_dir, normal_dir):
    """
    Post-processing step: Create visualizations for all generated map files
    Uses system Python environment rather than Blender's Python environment
    
    Args:
        base_dir: Base output directory
        depth_dir: Directory containing depth maps
        normal_dir: Directory containing normal maps
    """
    print("Starting visualization processing for all maps...")
    
    # Import visualization function
    sys.path.append(os.path.join(os.path.dirname(__file__), 'blender_script'))
    try:
        from visualize_normals import visualize_normal_map
        
        # Find all normal EXR files
        normal_exr_files = glob.glob(os.path.join(normal_dir, '*.exr'))
        
        if not normal_exr_files:
            print("Warning: No normal map EXR files found for visualization")
        else:    
            print(f"Found {len(normal_exr_files)} normal map files, starting visualization")
            
            # Process each normal map file
            for exr_file in normal_exr_files:
                # 提取纯数字文件名
                base_name = os.path.basename(exr_file)
                number_part = os.path.splitext(base_name)[0]  # 例如 '000'
                
                # 生成可视化文件的路径（在同一目录下）
                png_file = os.path.join(normal_dir, f"{number_part}_map.png")
                
                try:
                    print(f"Processing: {os.path.basename(exr_file)} → {os.path.basename(png_file)}")
                    visualize_normal_map(exr_file, png_file)
                except Exception as e:
                    print(f"Error visualizing file {exr_file}: {e}", file=sys.stderr)
        
        # Find all depth map files
        depth_files = glob.glob(os.path.join(depth_dir, '*.png'))
        if depth_files:
            print(f"Found {len(depth_files)} depth map files")
            # We don't need to process depth maps as they're already in PNG format
            # But we could apply additional visualization if needed
            for depth_file in depth_files:
                print(f"Depth map available: {os.path.basename(depth_file)}")
                
        print("All map visualizations completed!")
        
    except ImportError as e:
        print(f"Error: Cannot import visualize_normal_map function: {e}", file=sys.stderr)
        print("Please ensure required dependencies are installed: pip install OpenEXR numpy opencv-python", file=sys.stderr)

def _render(mesh_path, num_views=24, use_shader_ccm=True):
    """
    Main rendering function. Generates views, calls Blender for rendering each type of output.
    
    Args:
        mesh_path: Path to the 3D model file.
        num_views: Number of views to render.
        use_shader_ccm: Whether to use shader-based CCM generation (direct position-to-color).
    """
    # Setup output directories
    base_dir, camera_dir, rgb_dir, depth_dir, normal_dir, ccm_dir, face_id_dir, face_id_mapping_dir = setup_output_dirs()
    
    views = []

    # Random offset for Hammersley sequence
    offset = (np.random.rand(), np.random.rand())

    # Calculate optimal radius based on FOV
    fov_rad = np.pi / 4 # Field of view in radians
    diagonal = np.sqrt(3) # Diagonal of unit cube
    # Adjust radius slightly further than theoretical minimum for better framing
    optimal_radius = (diagonal / 2) / np.tan(fov_rad / 2) 
    radius = optimal_radius * 0.8
    
    for i in range(num_views):
        yaw_norm, pitch_norm = sphere_hammersley_sequence(i, num_views, offset)
        # Convert normalized angles to radians
        yaw = float(yaw_norm * 2 * np.pi) 
        pitch = float((pitch_norm - 0.5) * np.pi) # Map [0, 1] to [-pi/2, pi/2]
        
        extrinsic = get_camera_matrix(yaw, pitch, radius)
        views.append({
            "file_name": f"{i:03d}.png", # Keep png for reference, Blender script determines actual name
            "transform_matrix": extrinsic,
            "yaw": yaw,
            "pitch": pitch,
            "radius": radius,
            "fov": fov_rad # Store fov in radians
        })

    # Save camera parameters to camera directory
    transforms_path = os.path.join(camera_dir, 'transforms.json')
    with open(transforms_path, 'w') as f:
        json.dump(views, f, indent=2)
    print(f"Camera parameters have been saved to: {transforms_path}")

    # Batch views for rendering
    batch_size = 16 # Adjust based on memory/performance
    view_batches = [views[i:i + batch_size] for i in range(0, len(views), batch_size)]
    
    for batch_idx, batch_views in enumerate(view_batches):
        # Common parameters for all render calls
        common_args = [
            BLENDER_PATH, 
            '-b', # Background mode
            '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'), # Path to Blender script
            '--',
            '--views', json.dumps(batch_views), # Pass views as JSON string
            '--object', os.path.abspath(mesh_path), # Absolute path to mesh
            '--resolution', '512', # Image resolution
            '--engine', 'CYCLES', # Rendering engine
            '--batch_id', str(batch_idx) # Pass batch ID for file naming
        ]
        
        print(f"Starting Blender rendering batch {batch_idx + 1}/{len(view_batches)}...")
        
        # Use subprocess.run for better error handling and output capture
        import subprocess
        try:
            # Execute rendering command for RGB images
            rgb_args = common_args.copy()
            rgb_args.extend(['--output_folder', rgb_dir])
            print(f"  - Rendering RGB images...")
            subprocess.run(rgb_args, check=True, text=True)
            print(f"    RGB rendering complete")
            
            # Execute rendering command for normal maps
            normal_args = common_args.copy()
            normal_args.extend(['--output_folder', normal_dir, '--save_normal'])
            print(f"  - Rendering normal maps...")
            subprocess.run(normal_args, check=True, text=True)
            print(f"    Normal maps rendering complete")
            
            # Execute rendering command for depth maps
            depth_args = common_args.copy()
            depth_args.extend(['--output_folder', depth_dir, '--save_depth'])
            print(f"  - Rendering depth maps...")
            subprocess.run(depth_args, check=True, text=True)
            print(f"    Depth maps rendering complete")
            
            # Execute rendering command for CCM using shader-based method
            if use_shader_ccm:
                ccm_args = common_args.copy()
                ccm_args.extend(['--output_folder', ccm_dir, '--save_ccm'])
                print(f"  - Generating CCM directly using shader-based approach...")
                subprocess.run(ccm_args, check=True, text=True)
                print(f"    CCM generation complete")
            
            # Execute rendering command for Face ID
            face_id_args = common_args.copy()
            face_id_args.extend(['--output_folder', face_id_dir, '--save_face_id'])
            print(f"  - Generating Face ID maps...")
            subprocess.run(face_id_args, check=True, text=True)
            print(f"    Face ID map generation complete")
            
            print(f"Batch {batch_idx + 1} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Blender execution failed for batch {batch_idx + 1} with return code {e.returncode}.", file=sys.stderr)
            sys.exit(1) # Exit if Blender fails
        except FileNotFoundError:
            print(f"Error: Blender executable not found at '{BLENDER_PATH}'. Please check installation.", file=sys.stderr)
            sys.exit(1)

    print("All rendering batches completed.")
    
    # Visualize normal maps after rendering
    visualize_maps(base_dir, depth_dir, normal_dir)
    
    # Generate CCM images from depth maps (only if not using shader-based CCM)
    if not use_shader_ccm and batch_process_depth_to_ccm is not None:
        print("\nGenerating CCM (Canonical Coordinates Maps) from depth maps...")
        try:
            # Apply alpha mask from RGB images to remove background
            batch_process_depth_to_ccm(
                depth_dir, 
                ccm_dir, 
                transforms_path,
                apply_mask=True,
                rgb_dir=rgb_dir
            )
            print("CCM generation completed successfully!")
        except Exception as e:
            print(f"Error generating CCM maps: {e}", file=sys.stderr)
    else:
        print("Skipping post-process CCM generation since shader-based CCM was used.")
    
    print(f"""
Rendering complete! Output files are organized as follows:
- RGB images: {rgb_dir}/
- Depth maps: {depth_dir}/
- Normal maps: {normal_dir}/
- CCM (Canonical Coordinates Maps): {ccm_dir}/
- Face ID: {face_id_dir}/
- Face ID Mapping: {face_id_mapping_dir}/
- Camera parameters: {camera_dir}/transforms.json
""")

def main():
    parser = argparse.ArgumentParser(
        description="Render a 3D mesh from multiple spherical viewpoints and output images plus camera extrinsics."
    )
    parser.add_argument('--mesh', type=str, required=True,
                      help="Path to the 3D model file (e.g., .obj, .ply)")
    parser.add_argument('--num_views', type=int, default=24,
                      help="Number of views (images) to render")
    parser.add_argument('--use_shader_ccm', action='store_true', default=True,
                      help="Use shader-based CCM generation (faster and more accurate)")

    args = parser.parse_args()
    
    # Install Blender if needed
    print('Checking blender...', flush=True)
    _install_blender()
    
    # Execute rendering
    _render(args.mesh, args.num_views, args.use_shader_ccm)

if __name__ == '__main__':
    main()
