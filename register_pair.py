# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roy Amoyal, Oren Freifeld, Chaim Baskin

"""Register a pair of feature-3DGS models (source -> target).

Pipeline: optional perturbation of the source model -> semantic-weighted ICP
initialization -> differentiable feature-rendering refinement on the target's
cameras -> metrics + combined point clouds for inspection.

The registration machinery is imported from eval_register_3dgs_shapenet.py so
this script uses exactly the same math as the paper evaluation.

Example (two models trained with scripts/train_object.sh):
    python register_pair.py \
        --source_model output/my_car_B --target_model output/my_car_A \
        --target_data data/my_car_A --output_dir results/my_pair

Outputs in --output_dir:
    results.json         estimated transform (+ errors when GT is known)
    combined_icp.ply     target + source after the ICP initialization
    combined_final.ply   target + source after the full registration
    full_align.ply       the transformed source model alone
    (combined_gt.ply / combined_random.ply when a perturbation is applied)
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from scene import GaussianModel
from eval_register_3dgs_shapenet import (
    apply_transform_to_gaussian_model,
    compute_ate,
    compute_rre,
    create_random_transform,
    icp_weighted_semantic,
    save_combined_model,
    training,
)


def load_model(model_dir, iteration):
    ply_path = os.path.join(model_dir, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    if not os.path.exists(ply_path):
        raise SystemExit(f"Model ply not found: {ply_path}")
    gs = GaussianModel(3)
    gs.load_ply(ply_path)
    return gs


def main():
    parser = ArgumentParser(description="Register a pair of feature-3DGS models")
    parser.add_argument("--source_model", required=True,
                        help="Trained 3DGS model dir of the SOURCE object (the one that moves)")
    parser.add_argument("--target_model", required=True,
                        help="Trained 3DGS model dir of the TARGET object")
    parser.add_argument("--target_data", required=True,
                        help="The target object's data dir (images + transforms/sparse); "
                             "its cameras drive the refinement stage")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write results.json and the combined ply files")
    parser.add_argument("--images", default="train",
                        help="Images subdirectory name inside --target_data ('train' for "
                             "Blender-style data, 'images' for COLMAP data)")
    parser.add_argument("--iteration", type=int, default=7000,
                        help="Model iteration to load (point_cloud/iteration_<N>)")
    parser.add_argument("--icp_samples", type=int, default=30000,
                        help="Number of Gaussians sampled for the semantic ICP. Memory "
                             "scales quadratically: 30000 fits a 24 GB GPU, the paper's "
                             "ShapeNet evaluation used 50000 on a 48 GB GPU")
    parser.add_argument("--views_stride", type=int, default=-1,
                        help="Camera stride for the refinement stage; -1 (default) picks "
                             "5 evenly-spread diverse views")
    parser.add_argument("--random_transform", action="store_true",
                        help="Perturb the source with a random SE(3)+scale transform first "
                             "(evaluation setting; gives GT to measure errors against)")
    parser.add_argument("--transform_json", default=None,
                        help="Apply a known perturbation from a json file with keys "
                             "'transform' (4x4), 'rotation' (3x3), 'translation' (3), 'scale'")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    gs_target = load_model(args.target_model, args.iteration)
    gs_source = load_model(args.source_model, args.iteration)

    # Optional perturbation of the source (evaluation setting: the models are
    # already aligned and we measure how well the registration recovers the
    # inverse of the applied transform).
    T_gt = None
    transform_data = None
    if args.transform_json is not None:
        with open(args.transform_json, "r") as f:
            transform_data = json.load(f)
    elif args.random_transform:
        transform_data = create_random_transform(
            min_angle=45, max_angle=180,
            translation_range=1.0, min_scale=0.5, max_scale=1.5)

    if transform_data is not None:
        T_gt = np.array(transform_data["transform"])
        R_gt = torch.from_numpy(np.array(transform_data["rotation"])).float().cuda()
        t_gt = torch.from_numpy(np.array(transform_data["translation"])).float().cuda()
        s_gt = transform_data["scale"]

        save_combined_model(gs_target, gs_source,
                            os.path.join(args.output_dir, "combined_gt.ply"))
        gs_source = apply_transform_to_gaussian_model(gs_source, R_gt, t_gt, s_gt)
        save_combined_model(gs_target, gs_source,
                            os.path.join(args.output_dir, "combined_random.ply"))
        with open(os.path.join(args.output_dir, "applied_transform.json"), "w") as f:
            json.dump(transform_data, f, indent=4)

    # Stage 1: semantic-weighted ICP initialization
    print("Running semantic-weighted ICP initialization...")
    R_init, t_init, s_init = icp_weighted_semantic(
        gs_source.get_xyz, gs_source._semantic_feature,
        gs_target.get_xyz, gs_target._semantic_feature,
        n_samples=args.icp_samples)

    T_icp = np.eye(4)
    T_icp[:3, :3] = R_init.detach().cpu().numpy()
    T_icp[:3, 3] = t_init.detach().cpu().numpy()

    gs_icp = apply_transform_to_gaussian_model(
        gs_source, R_init.detach(), t_init.detach(), s_init.detach().item())
    save_combined_model(gs_target, gs_icp,
                        os.path.join(args.output_dir, "combined_icp.ply"))

    # Stage 2: differentiable feature-rendering refinement on the target cameras.
    # training() re-loads the source from a ply, so save the (possibly perturbed)
    # source into the output dir first.
    source_input_ply = os.path.join(args.output_dir, "source_input.ply")
    gs_source.save_ply(source_input_ply)

    final_results = training(
        target_model_path=args.target_model,
        source_model_path=args.source_model,
        target_data_path=args.target_data,
        parent_dir=None,
        initial_transform=(R_init.detach().cpu().numpy(),
                           t_init.detach().cpu().numpy(),
                           s_init.detach().cpu().numpy()),
        source_random_ply=source_input_ply,
        align_save_path=os.path.join(args.output_dir, "full_align.ply"),
        images=args.images,
        views_stride=args.views_stride,
        iteration=args.iteration,
    )

    gs_final = GaussianModel(3)
    gs_final.load_ply(final_results["transformed_model_path"])
    save_combined_model(gs_target, gs_final,
                        os.path.join(args.output_dir, "combined_final.ply"))

    results = {
        "source_model": args.source_model,
        "target_model": args.target_model,
        "T_icp": T_icp.tolist(),
        "T_est": final_results["T_est"].tolist(),
        "scale": final_results["scale"],
        "best_loss": float(final_results["best_loss"]),
    }
    if T_gt is not None:
        T_gt_inv = np.linalg.inv(T_gt)
        results["T_gt"] = T_gt.tolist()
        results["icp_ate"] = float(compute_ate(T_icp, T_gt_inv))
        results["icp_rre"] = float(compute_rre(T_icp, T_gt_inv))
        results["final_ate"] = float(compute_ate(final_results["T_est"], T_gt_inv))
        results["final_rre"] = float(compute_rre(final_results["T_est"], T_gt_inv))
        print(f"\nICP   - ATE: {results['icp_ate']:.4f}, RRE: {results['icp_rre']:.4f} deg")
        print(f"Final - ATE: {results['final_ate']:.4f}, RRE: {results['final_rre']:.4f} deg")

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nDone. Results written to {results_path}")


if __name__ == "__main__":
    main()
