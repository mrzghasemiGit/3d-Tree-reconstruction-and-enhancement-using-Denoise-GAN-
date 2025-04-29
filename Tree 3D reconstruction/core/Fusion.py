import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

def fuse_canopy_data(upper_path, lower_path):
    # Load biological point clouds
    canopy = o3d.io.read_point_cloud(upper_path)
    understory = o3d.io.read_point_cloud(lower_path)

    # Preprocess understory vegetation
    understory = understory.voxel_down_sample(voxel_size=0.02)
    understory, _ = understory.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Remove biological ground plane
    def remove_biological_ground(pcd, surface_threshold=0.1):
        ground_model, surface_points = pcd.segment_plane(surface_threshold, 3, 1000)
        return pcd.select_by_index(surface_points, invert=True)

    canopy_vegetation = remove_biological_ground(canopy)
    understory_vegetation = remove_biological_ground(understory)

    # Biological feature clustering
    def cluster_vegetation(data, eps=0.1, min_pts=10):
        labels = np.array(data.cluster_dbscan(eps=eps, min_points=min_pts))
        return [data.select_by_index(np.where(labels == i)[0]) for i in np.unique(labels) if i != -1]

    canopy_clusters = cluster_vegetation(canopy_vegetation)
    understory_clusters = cluster_vegetation(understory_vegetation)

    # Extract biological feature points
    def get_biological_features(clusters):
        return np.array([np.mean(cluster.points, axis=0)[:2] for cluster in clusters])

    canopy_features = get_biological_features(canopy_clusters)
    understory_features = get_biological_features(understory_clusters)

    # Compute biological descriptors
    def compute_biological_signatures(points):
        # Stem distance relationships
        D = cdist(points, points)
        quadratic_dist = D @ D.T
        
        # Crown structure ratios
        min_dists = np.min(quadratic_dist + np.eye(quadratic_dist.shape[0])*np.max(quadratic_dist), axis=1)
        crown_ratios = quadratic_dist / min_dists[:, None]
        
        # Branch angle relationships
        vectors = points[:, None] - points
        azimuth = np.arctan2(vectors[..., 1], vectors[..., 0])
        elevation = np.arctan2(vectors[..., 2], np.linalg.norm(vectors[..., :2], axis=-1))
        branch_angles = np.sqrt(azimuth**2 + elevation**2)
        
        return np.hstack([branch_angles.reshape(-1, branch_angles.shape[-1]), 
                        crown_ratios.reshape(-1, crown_ratios.shape[-1])])

    canopy_signature = compute_biological_signatures(canopy_features)
    understory_signature = compute_biological_signatures(understory_features)

    # Biological feature matching
    cost_matrix = np.linalg.norm(canopy_signature - understory_signature, axis=1)
    canopy_idx, understory_idx = linear_sum_assignment(cost_matrix)

    # Compute biological alignment
    canopy_pts = canopy_features[canopy_idx]
    understory_pts = understory_features[understory_idx]
    
    # Biological structure preservation
    canopy_center = np.mean(canopy_pts, axis=0)
    understory_center = np.mean(understory_pts, axis=0)
    
    H = (canopy_pts - canopy_center).T @ (understory_pts - understory_center)
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    scale = np.trace(S) / np.trace(canopy_pts.T @ canopy_pts)
    translation = understory_center - scale * rotation @ canopy_center

    # Fuse biological data
    understory.transform(np.vstack([np.hstack([scale*rotation, translation[:, None]]), [0, 0, 0, 1]])

    # Final fused biological model
    fused_model = canopy + understory
    o3d.io.write_point_cloud("fused_biological_model.ply", fused_model)
    
    # Visualize ecosystem structure
    o3d.visualization.draw_geometries([fused_model])

    return fused_model

# Biological data processing
fused_ecosystem = fuse_canopy_data("upper_canopy.ply", "lower_vegetation.ply")