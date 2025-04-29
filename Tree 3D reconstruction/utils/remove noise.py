import open3d as o3d
import argparse
import os
import sys

def process_point_cloud(input_path, output_path, voxel_size=0.09, 
                       nb_neighbors=30, std_ratio=0.5, visualize=False):
    """
    Process point cloud with voxel downsampling and statistical outlier removal
    """
    try:
        # Validate input path
        if not os.path.exists(input_path):
            raise FileNotFoundError("Input file not found")

        # Load data
        cloud = o3d.io.read_point_cloud(input_path)
        
        if visualize:
            o3d.visualization.draw_geometries([cloud], window_name="Input Cloud")

        # Downsample
        downsampled = cloud.voxel_down_sample(voxel_size)
        
        # Denoise
        clean_cloud, indices = downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
        # Save output
        o3d.io.write_point_cloud(output_path, clean_cloud)
        
        if visualize:
            o3d.visualization.draw_geometries([clean_cloud], window_name="Processed Cloud")

        print(f"Processed successfully: {len(clean_cloud.points)} points remaining")
        
    except Exception as e:
        sys.exit(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Point Cloud Processing Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True,
                       help="Input PLY file path")
    parser.add_argument("-o", "--output", required=True,
                       help="Output PLY file path")
    parser.add_argument("-v", "--voxel_size", type=float, default=0.09,
                       help="Voxel size for downsampling")
    parser.add_argument("-n", "--neighbors", type=int, default=30,
                       help="Number of neighbors for outlier removal")
    parser.add_argument("-s", "--std_ratio", type=float, default=0.5,
                       help="Standard deviation ratio threshold")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization windows")

    args = parser.parse_args()
    
    process_point_cloud(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        nb_neighbors=args.neighbors,
        std_ratio=args.std_ratio,
        visualize=args.visualize
    )