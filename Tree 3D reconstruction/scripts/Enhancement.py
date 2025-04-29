from core.DenoiseGAN import enhance_tree_structure

enhanced_tree = enhance_tree_structure(
    input_path="fused_canopy.ply",
    output_path="enhanced_tree.ply"
)