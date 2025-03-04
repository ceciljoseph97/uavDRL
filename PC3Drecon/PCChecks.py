import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
import copy

def compute_point_densities(pcd, k=10):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_points = np.asarray(pcd.points)
    num_points = len(pcd_points)
    densities = np.zeros(num_points)

    for i in tqdm(range(num_points), desc='Computing densities', unit='point'):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd_points[i], k)
        if len(idx) < k:
            densities[i] = 0
        else:
            k_nearest_points = pcd_points[idx[1:], :]
            distances = np.linalg.norm(k_nearest_points - pcd_points[i], axis=1)
            densities[i] = 1.0 / (np.mean(distances) + 1e-6)

    return densities

def print_density_statistics(densities, label):
    print(f"--- {label} ---")
    print(f"Minimum Density: {densities.min():.6f}")
    print(f"Maximum Density: {densities.max():.6f}")
    print(f"Mean Density: {densities.mean():.6f}")
    print(f"Median Density: {np.median(densities):.6f}")
    print(f"Standard Deviation: {densities.std():.6f}")
    print()

def get_filename(base_name, suffix, extension, output_dir, k, v):
    filename = f"{base_name}_{suffix}_k{k}_v{v}.{extension}"
    return os.path.join(output_dir, filename)

def compute_iou(pcd1_down, pcd2_down, voxel_size=0.1):
    print("Computing IoU...")

    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd1_down, voxel_size)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd2_down, voxel_size)

    voxels1 = set([tuple(voxel.grid_index) for voxel in voxel_grid1.get_voxels()])
    voxels2 = set([tuple(voxel.grid_index) for voxel in voxel_grid2.get_voxels()])

    intersection = voxels1.intersection(voxels2)
    union = voxels1.union(voxels2)

    iou = len(intersection) / len(union) if len(union) > 0 else 0.0

    print(f"IoU between the point clouds: {iou:.4f}\n")

    return iou

def generate_densities(pcd_path, voxel_size, k, output_dir):
    base_name = os.path.splitext(os.path.basename(pcd_path))[0]
    downsampled_pcd_filename = get_filename(
        base_name, "downsampled", "ply", output_dir, k, voxel_size)

    if os.path.exists(downsampled_pcd_filename):
        print(f"Loading downsampled point cloud from {downsampled_pcd_filename}...")
        pcd_down = o3d.io.read_point_cloud(downsampled_pcd_filename)
    else:
        print(f"Loading point cloud from {pcd_path}...")
        pcd = o3d.io.read_point_cloud(pcd_path)
        print("Downsampling point cloud...")
        pcd_down = pcd.voxel_down_sample(voxel_size)
        o3d.io.write_point_cloud(downsampled_pcd_filename, pcd_down)
        print(f"Downsampled point cloud saved to {downsampled_pcd_filename}\n")

    densities_filename = get_filename(
        base_name, "densities", "npy", output_dir, k, voxel_size)

    if os.path.exists(densities_filename):
        print(f"Loading precomputed densities from {densities_filename}...")
        densities = np.load(densities_filename)
    else:
        print("Computing densities...")
        densities = compute_point_densities(pcd_down, k)
        np.save(densities_filename, densities)
        print(f"Densities saved to {densities_filename}\n")

    return pcd_down, densities

def visualize_alignment(pcd1_down, pcd2_down, output_dir, base_name1,
                        base_name2, k, v):
    pcd1_temp = copy.deepcopy(pcd1_down)
    pcd2_temp = copy.deepcopy(pcd2_down)

    pcd1_temp.paint_uniform_color([0, 1, 0])
    pcd2_temp.paint_uniform_color([1, 0, 0])
    combined_pcd = pcd1_temp + pcd2_temp

    alignment_ply = get_filename(f"alignment_{base_name1}_vs_{base_name2}",
                                 "aligned", "ply", output_dir, k, v)
    o3d.io.write_point_cloud(alignment_ply, combined_pcd)
    print(f"Alignment point cloud saved to {alignment_ply}\n")
    print("Alignment visualization saved as a PLY file. You can view it using "
          "Open3D, MeshLab, or other 3D viewers.\n")

def generate_report(iou, reg_p2p, densities1_down, densities2_down,
                    mean_density_diff, median_density_diff, std_density_diff,
                    num_significant_differences, proportion_significant,
                    mapped_densities_stats, output_dir, base_name1, base_name2,
                    k, v):

    report_path = get_filename(f"comparison_report_{base_name1}_vs_{base_name2}",
                               "report", "txt", output_dir, k, v)
    with open(report_path, 'w') as f:
        f.write(f"Comparison Report between {base_name1} and {base_name2}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Voxel Size: {v}\n")
        f.write(f"Number of Nearest Neighbors (k): {k}\n\n")
        f.write(f"ICP Alignment Fitness: {reg_p2p.fitness:.4f}\n")
        f.write(f"ICP Inlier RMSE: {reg_p2p.inlier_rmse:.6f}\n\n")
        f.write("ICP Transformation Matrix:\n")
        f.write(f"{reg_p2p.transformation}\n\n")
        f.write(f"IoU: {iou:.4f}\n\n")
        f.write("--- Density Statistics ---\n")
        f.write(f"{base_name1}:\n")
        f.write(f"  Minimum Density: {densities1_down.min():.6f}\n")
        f.write(f"  Maximum Density: {densities1_down.max():.6f}\n")
        f.write(f"  Mean Density: {densities1_down.mean():.6f}\n")
        f.write(f"  Median Density: {np.median(densities1_down):.6f}\n")
        f.write(f"  Standard Deviation: {densities1_down.std():.6f}\n\n")
        f.write(f"{base_name2}:\n")
        f.write(f"  Minimum Density: {densities2_down.min():.6f}\n")
        f.write(f"  Maximum Density: {densities2_down.max():.6f}\n")
        f.write(f"  Mean Density: {densities2_down.mean():.6f}\n")
        f.write(f"  Median Density: {np.median(densities2_down):.6f}\n")
        f.write(f"  Standard Deviation: {densities2_down.std():.6f}\n\n")
        f.write("--- Mapped Densities Statistics ---\n")
        f.write(f"  Mean Mapped Density: {mapped_densities_stats['mean']:.6f}\n")
        f.write(f"  Median Mapped Density: {mapped_densities_stats['median']:.6f}\n")
        f.write(f"  Standard Deviation: {mapped_densities_stats['std']:.6f}\n\n")
        f.write("--- Density Difference Statistics ---\n")
        f.write(f"Mean Density Difference: {mean_density_diff:.6f}\n")
        f.write(f"Median Density Difference: {median_density_diff:.6f}\n")
        f.write(f"Standard Deviation of Density Differences: "
                f"{std_density_diff:.6f}\n\n")
        f.write(f"Number of Significant Differences: "
                f"{num_significant_differences}\n")
        f.write(f"Proportion of Significant Points: "
                f"{proportion_significant*100:.2f}%\n")
    print(f"Comparison report saved to {report_path}\n")

def compare_point_clouds(pcd_path1, pcd_path2, k=10, voxel_size=0.1,
                         output_dir='densityAnalysis'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name1 = os.path.splitext(os.path.basename(pcd_path1))[0]
    base_name2 = os.path.splitext(os.path.basename(pcd_path2))[0]

    pcd1_down, densities1_down = generate_densities(
        pcd_path1, voxel_size, k, output_dir)
    pcd2_down, densities2_down = generate_densities(
        pcd_path2, voxel_size, k, output_dir)

    print_density_statistics(densities1_down, f"Point Cloud 1 ({base_name1})")
    print_density_statistics(densities2_down, f"Point Cloud 2 ({base_name2})")

    print("Aligning point clouds...")
    threshold = voxel_size * 1.5
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2_down, pcd1_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print(f"ICP Fitness: {reg_p2p.fitness:.4f}")
    print(f"ICP Inlier RMSE: {reg_p2p.inlier_rmse:.6f}\n")

    alignment_quality_threshold = 0.5
    if reg_p2p.fitness > alignment_quality_threshold:
        aligned_pcd2_filename = get_filename(
            base_name2, f"aligned_to_{base_name1}", "ply", output_dir, k,
            voxel_size)
        if os.path.exists(aligned_pcd2_filename):
            print(f"Loading already aligned point cloud from "
                  f"{aligned_pcd2_filename}...\n")
            pcd2_down = o3d.io.read_point_cloud(aligned_pcd2_filename)
        else:
            pcd2_down.transform(reg_p2p.transformation)
            o3d.io.write_point_cloud(aligned_pcd2_filename, pcd2_down)
            print(f"Point clouds aligned successfully and saved to "
                  f"{aligned_pcd2_filename}\n")
    else:
        print("Warning: Alignment quality is low. IoU may not be reliable.\n")

    visualize_alignment(pcd1_down, pcd2_down, output_dir, base_name1,
                        base_name2, k, voxel_size)

    iou = compute_iou(pcd1_down, pcd2_down, voxel_size=voxel_size)

    print("Mapping densities between point clouds...")
    mapped_densities_filename = get_filename(
        f"{base_name1}_mapped_to_{base_name2}", "mapped_densities", "npy",
        output_dir, k, voxel_size)
    if os.path.exists(mapped_densities_filename):
        print(f"Loading mapped densities from {mapped_densities_filename}...\n")
        mapped_densities2 = np.load(mapped_densities_filename)
    else:
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2_down)
        mapped_densities2 = []
        for point in tqdm(np.asarray(pcd1_down.points), desc='Mapping densities',
                          unit='point'):
            [_, idx, _] = pcd2_tree.search_knn_vector_3d(point, 1)
            mapped_densities2.append(densities2_down[idx[0]])
        mapped_densities2 = np.array(mapped_densities2)
        np.save(mapped_densities_filename, mapped_densities2)
        print(f"Mapped densities saved to {mapped_densities_filename}\n")

    print("Computing density differences...")
    density_diff_filename = get_filename(
        f"density_diff_{base_name1}_{base_name2}", "density_diff", "npy",
        output_dir, k, voxel_size)
    if os.path.exists(density_diff_filename):
        print(f"Loading density differences from {density_diff_filename}...\n")
        density_diff = np.load(density_diff_filename)
    else:
        density_diff = densities1_down - mapped_densities2
        np.save(density_diff_filename, density_diff)
        print(f"Density differences saved to {density_diff_filename}\n")

    print("Creating density difference point cloud...")
    # Normalize density_diff for color mapping
    density_diff_normalized = (
        (density_diff - density_diff.min()) /
        (density_diff.max() - density_diff.min() + 1e-6)
    )

    colormap = plt.get_cmap('coolwarm')
    colors_rgb = colormap(density_diff_normalized)[:, :3]  # RGB

    pcd1_down.colors = o3d.utility.Vector3dVector(colors_rgb)

    density_diff_ply = get_filename(
        f"density_diff_{base_name1}_{base_name2}", "density_diff", "ply",
        output_dir, k, voxel_size)
    o3d.io.write_point_cloud(density_diff_ply, pcd1_down)
    print(f"Density differences point cloud saved to {density_diff_ply}\n")

    print("Identifying significant differences...")
    density_diff_std = density_diff.std()
    threshold_diff = density_diff_std * 2
    significant_indices = np.where(np.abs(density_diff) > threshold_diff)[0]
    significant_points = pcd1_down.select_by_index(significant_indices)
    significant_points.paint_uniform_color([1, 0, 0])
    remaining_points = pcd1_down.select_by_index(significant_indices,
                                                 invert=True)
    remaining_points.paint_uniform_color([0.7, 0.7, 0.7])

    # Save significant differences point cloud
    significant_diff_ply = get_filename(
        f"significant_differences_{base_name1}_{base_name2}",
        "significant_diff", "ply", output_dir, k, voxel_size)
    combined_pcd = remaining_points + significant_points
    o3d.io.write_point_cloud(significant_diff_ply, combined_pcd)
    print(f"Significant differences point cloud saved to {significant_diff_ply}\n")

    mean_density_diff = density_diff.mean()
    median_density_diff = np.median(density_diff)
    std_density_diff = density_diff_std
    num_significant_differences = len(significant_indices)
    total_points = len(density_diff)
    proportion_significant = num_significant_differences / total_points

    mapped_densities_stats = {
        'mean': mapped_densities2.mean(),
        'median': np.median(mapped_densities2),
        'std': mapped_densities2.std()
    }

    generate_report(iou, reg_p2p, densities1_down, densities2_down,
                    mean_density_diff, median_density_diff, std_density_diff,
                    num_significant_differences, proportion_significant,
                    mapped_densities_stats, output_dir, base_name1, base_name2,
                    k, voxel_size)

def main():
    parser = argparse.ArgumentParser(
        description="Generate densities or compare point clouds.")
    parser.add_argument("mode", choices=["generate", "compare"],
                        help="Mode of operation: 'generate' or 'compare'.")
    parser.add_argument("pointcloud1",
                        help="Path to the first point cloud file (e.g., .pcd, .ply).")
    parser.add_argument("pointcloud2", nargs='?', default=None,
                        help="Path to the second point cloud file for comparison.")
    parser.add_argument("-k", "--neighbors", type=int, default=10,
                        help="Number of nearest neighbors for density estimation "
                             "(default: 10).")
    parser.add_argument("-v", "--voxel_size", type=float, default=0.1,
                        help="Voxel size for downsampling (default: 0.1).")
    parser.add_argument("-o", "--output_dir", type=str, default="densityAnalysis",
                        help="Directory to save output files "
                             "(default: 'densityAnalysis').")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_densities(args.pointcloud1, args.voxel_size, args.neighbors,
                           args.output_dir)
    elif args.mode == "compare":
        if not args.pointcloud2:
            print("Error: You must provide two point clouds for comparison.")
            sys.exit(1)
        compare_point_clouds(args.pointcloud1, args.pointcloud2,
                             k=args.neighbors, voxel_size=args.voxel_size,
                             output_dir=args.output_dir)
    else:
        print("Invalid mode. Use 'generate' or 'compare'.")
        sys.exit(1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
