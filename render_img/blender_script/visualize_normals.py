# visualize_normals.py
import argparse # Keep for potential future CLI usage or example
import sys
import os
import numpy as np
import cv2
# import matplotlib.pyplot as plt # No longer needed if not showing plot
# Add imports for OpenEXR and Imath
try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: OpenEXR library is missing. Please run 'pip install OpenEXR'", file=sys.stderr)
    # If used as a module, raising might be better than exiting
    raise ImportError("OpenEXR library is missing. Please run 'pip install OpenEXR'")

def visualize_normal_map(exr_path: str, output_path: str):
    """
    Reads a normal map from an EXR file, visualizes it, and saves the result
    to the specified output path.

    Args:
        exr_path (str): Path to the input EXR normal map file.
        output_path (str): Path to save the visualized image file (e.g., 'output.png').
                           The file format is determined by the extension (supported by cv2.imwrite).
    Raises:
        FileNotFoundError: If the input exr_path file does not exist.
        ValueError: If the EXR file cannot be read or does not contain expected channels.
        IOError: If the output file cannot be written.
    """
    # --- Input Validation ---
    if not os.path.isfile(exr_path):
        raise FileNotFoundError(f"Input file not found: {exr_path}")
    if not output_path:
         raise ValueError("The output_path argument must be provided.")

    # --- Start: EXR Reading Logic ---
    normal_map = None
    try:
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Check available channels
        channels = header['channels'].keys()
        # print(f"Available channels: {list(channels)}") # Can be commented out for module usage

        # Determine channels to read (usually R, G, B)
        channel_map = {}
        if 'R' in channels and 'G' in channels and 'B' in channels:
            channel_map = {'R': 'R', 'G': 'G', 'B': 'B'}
        elif 'Z' in channels: # Handle depth map or single channel case
             print("Warning: Detected Z channel, possibly a depth map or grayscale image. Will replicate to 3 channels for visualization.", file=sys.stderr)
             pt = Imath.PixelType(Imath.PixelType.FLOAT)
             z_str = exr_file.channel('Z', pt)
             img = np.frombuffer(z_str, dtype=np.float32)
             img.shape = (size[1], size[0]) # Single channel
             exr_file.close()
             # Replicate single channel to three channels for visualization
             normal_map = np.stack([img]*3, axis=-1)
        else:
            exr_file.close()
            raise ValueError(f"Error: Could not find standard R, G, B, or Z channels in EXR file {exr_path}.")

        # If normal_map was not assigned by Z channel logic
        if normal_map is None:
            # Read RGB channels
            pt = Imath.PixelType(Imath.PixelType.FLOAT) # Normal maps typically use float
            red_str = exr_file.channel(channel_map['R'], pt)
            green_str = exr_file.channel(channel_map['G'], pt)
            blue_str = exr_file.channel(channel_map['B'], pt)

            # Convert channel data to NumPy arrays
            red = np.frombuffer(red_str, dtype=np.float32)
            green = np.frombuffer(green_str, dtype=np.float32)
            blue = np.frombuffer(blue_str, dtype=np.float32)

            red.shape = green.shape = blue.shape = (size[1], size[0]) # H, W

            # Combine channels H x W x C (BGR order for cv2.imwrite compatibility)
            normal_map = np.stack([blue, green, red], axis=-1) # BGR for OpenCV compatibility
            exr_file.close()

    except Exception as e:
        raise ValueError(f"Error reading EXR file {exr_path}: {e}")
    # --- End: EXR Reading Logic ---

    if normal_map is None: # Safeguard
        raise ValueError(f"Failed to generate normal map data after processing EXR file {exr_path}.")

    # Expect H x W x 3 image
    if normal_map.ndim != 3 or normal_map.shape[2] != 3:
         raise ValueError(f"Error: Expected a 3-channel image, but got shape {normal_map.shape}")

    # --- Start: Normal Map Processing --- 
    # print(f"Normal map dtype: {normal_map.dtype}, shape: {normal_map.shape}, min: {np.min(normal_map):.2f}, max: {np.max(normal_map):.2f}") # Optional debug print

    # Map range
    if np.min(normal_map) >= -0.01 and np.max(normal_map) <= 1.01:
        # print("Normal map appears to be in [0, 1] range.")
         normal_viz = normal_map
    elif np.min(normal_map) >= -1.01 and np.max(normal_map) <= 1.01:
        # print("Normal map appears to be in [-1, 1] range. Remapping to [0, 1].")
         normal_viz = (normal_map * 0.5 + 0.5)
    else:
        print(f"Warning: Normal map values (range [{np.min(normal_map):.2f}, {np.max(normal_map):.2f}]) outside expected [-1, 1] or [0, 1]. Applying min-max normalization.", file=sys.stderr)
        min_val = np.min(normal_map)
        max_val = np.max(normal_map)
        if max_val > min_val:
            normal_viz = (normal_map - min_val) / (max_val - min_val)
        else:
            # Avoid division by zero if all values are the same
            normal_viz = np.zeros_like(normal_map)

    # Clip range and convert to 8-bit
    normal_viz = np.clip(normal_viz, 0, 1)
    normal_viz_8bit = (normal_viz * 255).astype(np.uint8)
    # --- End: Normal Map Processing --- 

    # --- Start: Saving Output --- 
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if directory is not empty (e.g., just a filename)
             os.makedirs(output_dir, exist_ok=True)

        # Save the 8-bit BGR image using cv2.imwrite
        success = cv2.imwrite(output_path, normal_viz_8bit)
        if not success:
            raise IOError(f"cv2.imwrite failed to save file to {output_path}")
        else:
            print(f"Visualization successfully saved to: {output_path}") # Confirmation message

    except Exception as e:
        raise IOError(f"Could not save visualization to {output_path}: {e}")
    # --- End: Saving Output --- 

# --- Example Usage Block --- 
if __name__ == '__main__':
    # This part executes only when the script is run directly, for demonstration or testing
    parser = argparse.ArgumentParser(
        description='Visualize an EXR normal map and save it as an image file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualize_normals.py input.exr output.png
"""
    )
    parser.add_argument('input_exr', type=str, help='Path to the input EXR normal map file.')
    parser.add_argument('output_image', type=str, help='Path to save the output visualization image (e.g., output.png).')

    # Check if running inside Blender environment
    try:
        import bpy
        # If inside Blender, do not parse command line arguments
        print("Warning: This script should not be run directly inside Blender. Use it as a standalone Python module.", file=sys.stderr)
        # Optionally raise an error or exit here
        # sys.exit(1)
    except ImportError:
        # Parse arguments only if outside Blender environment
        if len(sys.argv) > 1: # Check if command line arguments were provided
            args = parser.parse_args()
            try:
                # Call the main function
                visualize_normal_map(args.input_exr, args.output_image)
                print("Script finished successfully.")
            except (FileNotFoundError, ValueError, IOError) as e:
                print(f"Error during processing: {e}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # If no arguments, print help or a message
            print("Please provide input EXR file and output image file paths. Use -h for help.")
            # parser.print_help() # Can uncomment to show help

    # You can also call the function directly here for testing without command-line args:
    # test_exr = 'render_img/render_output/normal_map/000_normal.exr'
    # test_out = 'render_img/render_output/normal_map/test_visualization.png'
    # if os.path.exists(test_exr):
    #     try:
    #         print(f"\nTest run: {test_exr} -> {test_out}")
    #         visualize_normal_map(test_exr, test_out)
    #     except Exception as e:
    #         print(f"Error during test run: {e}")
    # else:
    #     print(f"Test file {test_exr} not found, skipping test run.")