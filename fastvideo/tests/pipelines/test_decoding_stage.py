from types import SimpleNamespace

import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage


class _RecordingDecodingStage(DecodingStage):

    def __init__(self):
        super().__init__(vae=object())
        self.decode_calls = 0

    def decode(self, latents: torch.Tensor, fastvideo_args) -> torch.Tensor:
        self.decode_calls += 1
        return latents + 1


def _args(output_type: str = "pil"):
    return SimpleNamespace(
        output_type=output_type,
        model_loaded={"vae": True},
        vae_cpu_offload=False,
    )


def _batch(*, save_video: bool, return_frames: bool) -> ForwardBatch:
    return ForwardBatch(
        data_type="video",
        latents=torch.ones(1, 3, 2, 4, 4),
        save_video=save_video,
        return_frames=return_frames,
    )


def test_decoding_stage_skips_pixel_decode_for_no_output_request():
    stage = _RecordingDecodingStage()
    batch = _batch(save_video=False, return_frames=False)

    result = stage.forward(batch, _args())

    assert stage.decode_calls == 0
    assert result.extra["pixel_decode_skipped"] is True
    torch.testing.assert_close(result.output, batch.latents)


def test_decoding_stage_decodes_when_frames_are_returned():
    stage = _RecordingDecodingStage()
    batch = _batch(save_video=False, return_frames=True)

    result = stage.forward(batch, _args())

    assert stage.decode_calls == 1
    assert "pixel_decode_skipped" not in result.extra
    torch.testing.assert_close(result.output, batch.latents + 1)


def test_decoding_stage_decodes_when_video_is_saved():
    stage = _RecordingDecodingStage()
    batch = _batch(save_video=True, return_frames=False)

    result = stage.forward(batch, _args())

    assert stage.decode_calls == 1
    assert "pixel_decode_skipped" not in result.extra
    torch.testing.assert_close(result.output, batch.latents + 1)
