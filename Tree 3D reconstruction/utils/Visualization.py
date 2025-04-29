import open3d as o3d

tree = o3d.io.read_point_cloud("data/output/enhanced_tree.ply")
o3d.visualization.draw_geometries([tree])