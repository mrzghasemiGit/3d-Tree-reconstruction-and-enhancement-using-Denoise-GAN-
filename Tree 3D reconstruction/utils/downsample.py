import open3d as o3d
import argparse
import os
import sys

def downsample_point_cloud(input_path, output_path, voxel_size=0.4, visualize=False):
    """
    Downsample point cloud using voxel grid method
    :param input_path: Input PLY file path
    :param output_path: Output PLY file path
    :param voxel_size: Grid size for downsampling (in meters)
    :param visualize: Show visualization windows
    """
    try:
        # Validate paths
        if not os.path.exists(input_path):
            raise FileNotFoundError("Input file does not exist")
            
        if os.path.abspath(input_path) == os.path.abspath(output_path):
            raise ValueError("Input and output paths must be different")

        # Load data
        cloud = o3d.io.read_point_cloud(input_path)
        
        if visualize:
            o3d.visualization.draw_geometries([cloud], window_name="Original Cloud")

        # Process data
        downsampled = cloud.voxel_down_sample(voxel_size)
        
        # Save result
        o3d.io.write_point_cloud(output_path, downsampled)
        
        if visualize:
            o3d.visualization.draw_geometries([downsampled], window_name="Downsampled Cloud")

        print(f"Downsampling complete: {len(downsampled.points)} points remaining")
        
    except Exception as e:
        sys.exit(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Point Cloud Downsampling Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True,
                       help="Input PLY file path")
    parser.add_argument("-o", "--output", required=True,
                       help="Output PLY file path")
    parser.add_argument("-v", "--voxel_size", type=float, default=0.4,
                       help="Voxel grid size for downsampling")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization windows")

    args = parser.parse_args()
    
    downsample_point_cloud(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        visualize=args.visualize
    )