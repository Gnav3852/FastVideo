from pathlib import Path
import argparse
import time

import torch

from fastvideo import VideoGenerator


PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs_video/ltx2_decode_profile")
    parser.add_argument("--profile-name", default="default")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-frames", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def run_generation(generator, args, output_path: str, save_video: bool):
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_output = str(output_dir / f"warmup_{args.profile_name}.mp4")
    final_output = str(output_dir / f"ltx2_{args.profile_name}.mp4")

    print("Config:")
    print(f"  frames={args.num_frames}")
    print(f"  height={args.height}")
    print(f"  width={args.width}")
    print(f"  steps={args.num_inference_steps}")
    print(f"  warmups={args.warmup_runs}")
    print(f"  save_video={args.save_video}")
    print(f"  return_frames={args.return_frames}")

    with torch.cuda.nvtx.range("ltx2:model_load"):
        load_start = time.perf_counter()
        generator = VideoGenerator.from_pretrained(
            "Davids048/LTX2-Base-Diffusers",
            num_gpus=1,
        )
        torch.cuda.synchronize()
        load_s = time.perf_counter() - load_start

    print(f"Model load seconds: {load_s:.3f}")

    print(f"Running {args.warmup_runs} warmup runs...")
    for i in range(args.warmup_runs):
        print(f"Warmup {i + 1}/{args.warmup_runs}")
        with torch.cuda.nvtx.range(f"ltx2:warmup_{i + 1}"):
            run_generation(generator, args, output_path=warmup_output, save_video=False)
            torch.cuda.synchronize()

    print("Warmup complete.")
    print("Starting profiled inference...")

    with torch.cuda.nvtx.range(f"ltx2:profiled_generate:{args.profile_name}"):
        start = time.perf_counter()
        run_generation(generator, args, output_path=final_output, save_video=args.save_video)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    print(f"Profiled inference seconds: {elapsed:.3f}")
    print("Inference complete.")

    generator.shutdown()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
