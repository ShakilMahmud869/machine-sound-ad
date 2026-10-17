"""Correctness checks for the feature pipeline. Run: python -m pytest"""
import pytest


@pytest.mark.skip(reason="TODO: remove once stack_frames is implemented")
def test_stack_frames_matches_loop():
    # TODO: random spec of shape (128, 50); assert
    # torch.equal(stack_frames(spec, 5), stack_frames_loop(spec, 5))
    # and that the shape is (46, 640).
    pass


@pytest.mark.skip(reason="TODO: remove once logmel is implemented")
def test_logmel_is_finite():
    # TODO: synthesise 1 s of noise with torch, save to a tmp WAV
    # (pytest's tmp_path fixture), run logmel, assert torch.isfinite(...).all()
    pass
