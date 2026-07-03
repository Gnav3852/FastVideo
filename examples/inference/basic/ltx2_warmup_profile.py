import argparse
import time
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

from fastvideo import VideoGenerator


PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warmup-aware LTX2 profiling script for DGX Spark.")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_video/ltx2_basic"))
    parser.add_argument("--save-video", action="store_true", help="Save the measured run as mp4.")
    parser.add_argument("--return-frames", action="store_true", help="Return decoded frames from each run.")
    parser.add_argument(
        "--output-type",
        choices=("pil", "latent"),
        default="pil",
        help="Use latent output to skip VAE pixel decode.",
    )
    parser.add_argument("--profile-name", default=None, help="NVTX label suffix for the measured run.")
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_generation(generator: VideoGenerator, args: argparse.Namespace, output_path: str, save_video: bool) -> None:
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=save_video,
        return_frames=args.return_frames,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
    )


def timed_generation(
    generator: VideoGenerator,
    args: argparse.Namespace,
    label: str,
    output_path: str,
    save_video: bool,
) -> float:
    nvtx.range_push(label)
    start = time.perf_counter()
    try:
        run_generation(generator, args, output_path=output_path, save_video=save_video)
        synchronize()
    finally:
        nvtx.range_pop()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return elapsed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    warmup_output = str(args.output_dir / "warmup.mp4")
    final_output = str(args.output_dir / "ltx2_trace_test.mp4")
    measured_suffix = args.profile_name or (
        f"{args.output_type}_save_{args.save_video}_return_{args.return_frames}"
    )
    measured_label = f"ltx2_profiled_generate:{measured_suffix}"

    print("Loading LTX2 generator...")
    nvtx.range_push("ltx2_load_generator")
    load_start = time.perf_counter()
    try:
        generator = VideoGenerator.from_pretrained(
            "Davids048/LTX2-Base-Diffusers",
            num_gpus=1,
            output_type=args.output_type,
        )
        synchronize()
    finally:
        nvtx.range_pop()
    print(f"ltx2_load_generator: {time.perf_counter() - load_start:.3f}s")

    print(f"Running {args.warmup_runs} warmup runs...")
    for i in range(args.warmup_runs):
        timed_generation(
            generator,
            args,
            label=f"ltx2_warmup_{i + 1}",
            output_path=warmup_output,
            save_video=False,
        )
    print("Warmup complete.")

    print(
        "Starting measured inference: "
        f"save_video={args.save_video}, return_frames={args.return_frames}, "
        f"output_type={args.output_type}"
    )
    timed_generation(
        generator,
        args,
        label=measured_label,
        output_path=final_output,
        save_video=args.save_video,
    )
    print("Measured inference complete.")

    nvtx.range_push("ltx2_shutdown")
    shutdown_start = time.perf_counter()
    try:
        generator.shutdown()
        synchronize()
    finally:
        nvtx.range_pop()
    print(f"ltx2_shutdown: {time.perf_counter() - shutdown_start:.3f}s")


if __name__ == "__main__":
    main()
