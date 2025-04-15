# import torch
# import numpy as np
# import trimesh
# import imageio
# import argparse
# import os
# import sys

# # Add the project root to the Python path to allow importing project modules
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# try:
#     from preview.utils import render_utils
#     from preview.utils.representations import MeshExtractResult
# except ImportError as e:
#     print(f"Error importing project modules: {e}")
#     print("Please ensure you are running this script from the 'tools' directory or have added the project root to your PYTHONPATH.")
#     sys.exit(1)

# def main(args):
#     if not torch.cuda.is_available():
#         print("CUDA is not available. This script requires a GPU.")
#         sys.exit(1)
#     device = torch.device("cuda")

#     # 1. Load the mesh
#     try:
#         mesh = trimesh.load_mesh(args.input_mesh)
#         print(f"Loaded mesh: {args.input_mesh} (Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)})")
#     except Exception as e:
#         print(f"Error loading mesh file {args.input_mesh}: {e}")
#         sys.exit(1)

#     # Ensure the mesh has vertices and faces
#     if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
#          print(f"Mesh '{args.input_mesh}' does not contain vertices or faces.")
#          sys.exit(1)

#     # Convert to tensors
#     vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
#     faces_tensor = torch.tensor(mesh.faces, dtype=torch.int32, device=device) # MeshRenderer expects int32

#     # Normalize vertices to fit within a unit cube centered at origin if desired
#     # (This often helps with consistent rendering camera parameters)
#     # center = vertices_tensor.mean(0)
#     # vertices_tensor -= center
#     # scale = torch.max(torch.sqrt(torch.sum(vertices_tensor**2, dim=1)))
#     # vertices_tensor /= scale
#     # print(f"Normalized mesh vertices.") # Uncomment if normalization is used


#     # 2. Prepare input for renderer
#     try:
#         mesh_input = MeshExtractResult(vertices=vertices_tensor, faces=faces_tensor)
#         if not mesh_input.success:
#              print("Failed to create MeshExtractResult (likely due to empty vertices/faces after tensor conversion).")
#              sys.exit(1)
#     except Exception as e:
#         print(f"Error creating MeshExtractResult: {e}")
#         sys.exit(1)


#     # 3. Render Normal Map Video
#     print(f"Rendering normal map video with {args.num_frames} frames...")
#     try:
#         render_output = render_utils.render_video(
#             mesh_input,
#             num_frames=args.num_frames,
#             resolution=args.resolution,
#             r=args.camera_distance, # Use arg for camera distance
#             fov=args.camera_fov,     # Use arg for field of view
#             verbose=True             # Show progress bar
#         )
#     except Exception as e:
#         print(f"Error during rendering: {e}")
#         sys.exit(1)

#     # 4. Extract Normal Frames
#     if 'normal' not in render_output or not render_output['normal']:
#         print("Error: 'normal' key not found or empty in render output.")
#         print("Available keys:", render_output.keys())
#         # Check if the MeshRenderer actually produced normal output
#         # This might happen if the mesh data is degenerate or rendering options are incorrect
#         sys.exit(1)

#     normal_frames = render_output['normal']
#     print(f"Generated {len(normal_frames)} normal map frames.")

#     # 5. Save Video
#     try:
#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
#         imageio.mimsave(args.output_video, normal_frames, fps=args.fps)
#         print(f"Successfully saved normal map video to: {args.output_video}")
#     except Exception as e:
#         print(f"Error saving video: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Render a normal map video from a 3D mesh file.")
#     parser.add_argument("input_mesh", type=str, help="Path to the input mesh file (e.g., .obj, .ply).")
#     parser.add_argument("output_video", type=str, help="Path to save the output MP4 video file.")
#     parser.add_argument("--num_frames", type=int, default=120, help="Number of frames for the output video.")
#     parser.add_argument("--resolution", type=int, default=512, help="Resolution of the output video (width and height).")
#     parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
#     parser.add_argument("--camera_distance", type=float, default=2.0, help="Distance of the camera from the origin.")
#     parser.add_argument("--camera_fov", type=float, default=40.0, help="Camera field of view in degrees.")


#     args = parser.parse_args()
#     main(args) 