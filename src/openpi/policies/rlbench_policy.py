import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_rlbench_example() -> dict:
    """Creates a random input example for the RLBench policy."""
    return {
        "observation/front": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_to_rot6d(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Converts an xyzw quaternion into the 6D rotation representation."""
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float32)
    if quaternion_xyzw.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {quaternion_xyzw.shape}.")

    norm = np.linalg.norm(quaternion_xyzw, axis=-1, keepdims=True)
    quaternion_xyzw = quaternion_xyzw / np.clip(norm, 1e-8, None)
    x, y, z, w = np.moveaxis(quaternion_xyzw, -1, 0)

    # Convert xyzw quaternion to a rotation matrix, then keep the first two columns.
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    matrix = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(*quaternion_xyzw.shape[:-1], 3, 3)

    return matrix[..., :, :2].reshape(*quaternion_xyzw.shape[:-1], 6).astype(np.float32)


def _pose8_to_pose10(pose8: np.ndarray) -> np.ndarray:
    """Converts [xyz, quat_xyzw, grip] to [xyz, rot6d, grip]."""
    pose8 = np.asarray(pose8, dtype=np.float32)
    if pose8.shape[-1] != 8:
        raise ValueError(f"Expected pose last dim 8, got {pose8.shape}.")

    xyz = pose8[..., :3]
    quat = pose8[..., 3:7]
    grip = pose8[..., 7:8]
    return np.concatenate([xyz, _quat_to_rot6d(quat), grip], axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class RLBenchInputs(transforms.DataTransformFn):
    """Maps RLBench observations/actions into the pi0/pi0.5 input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # The M0 baseline follows the official single-arm setup:
        # front -> base, wrist -> left_wrist, right_wrist -> zero padding.
        base_image = _parse_image(data["observation/front"])
        wrist_image = _parse_image(data["observation/wrist"])

        inputs = {
            "state": _pose8_to_pose10(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # During training, convert the raw 8D RLBench action chunk into 10D as well.
        if "actions" in data:
            inputs["actions"] = _pose8_to_pose10(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RLBenchOutputs(transforms.DataTransformFn):
    """Returns the unpadded 10D action representation used by RLBench."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :10])}
