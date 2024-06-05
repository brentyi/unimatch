from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from unimatch.unimatch import UniMatch


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (
            ((self.ht // padding_factor) + 1) * padding_factor - self.ht
        ) % padding_factor
        pad_wd = (
            ((self.wd // padding_factor) + 1) * padding_factor - self.wd
        ) % padding_factor
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def run_optical_flow(image0: Path, image1: Path, /) -> None:
    device = torch.device("cuda")

    # Load model.
    model = UniMatch(
        num_scales=2,
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        upsample_factor=4,
        task="flow",
    )
    checkpoint = torch.load(
        "./gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device=device)

    # Read the input images.
    img0_np = iio.imread(image0)[800:]
    img1_np = iio.imread(image1)[800:]

    # Important: model takes BCHW uint8 images, 0-255,
    img0_torch = torch.from_numpy(img0_np).float().to(device=device)
    img1_torch = torch.from_numpy(img1_np).float().to(device=device)
    img0_torch = img0_torch.permute(2, 0, 1)[None, :, :, :]
    img1_torch = img1_torch.permute(2, 0, 1)[None, :, :, :]
    assert img0_torch.shape == img1_torch.shape

    # The UniFlow training assumes width > height.
    transpose_img = False
    if img0_torch.shape[-2] > img0_torch.shape[-1]:
        img0_torch = torch.transpose(img0_torch, -2, -1)
        img1_torch = torch.transpose(img1_torch, -2, -1)
        transpose_img = True

    padding_factor = 32
    max_inference_size = [384, 768]
    nearest_size = [
        int(np.ceil(img0_torch.shape[-2] / padding_factor)) * padding_factor,
        int(np.ceil(img0_torch.shape[-1] / padding_factor)) * padding_factor,
    ]
    inference_size = [
        min(max_inference_size[0], nearest_size[0]),
        min(max_inference_size[1], nearest_size[1]),
    ]
    ori_size = img0_torch.shape[-2:]
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        img0_torch = F.interpolate(
            img0_torch, size=inference_size, mode="bilinear", align_corners=True
        )
        img1_torch = F.interpolate(
            img1_torch, size=inference_size, mode="bilinear", align_corners=True
        )

    with torch.inference_mode():
        results = model.forward(
            img0_torch,
            img1_torch,
            attn_type="swin",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
        )
        flow_preds = results["flow_preds"]
        assert isinstance(flow_preds, list)
    final_flow = flow_preds[-1]

    # Upsample the flow.
    final_flow = F.interpolate(
        final_flow, size=(ori_size[0], ori_size[1]), mode="bilinear", align_corners=True
    )
    final_flow[:, 0, :, :] *= ori_size[0] / inference_size[0]
    final_flow[:, 1, :, :] *= ori_size[1] / inference_size[1]
    assert final_flow.shape == (1, 2, ori_size[0], ori_size[1])

    # Untranspose the flow.
    if transpose_img:
        final_flow = torch.transpose(final_flow, -2, -1)
        final_flow = torch.stack(
            [
                final_flow[:, 1, :, :],
                final_flow[:, 0, :, :],
            ],
            dim=1,
        )

    visualize_optical_flow(final_flow)

    import matplotlib.pyplot as plt

    (orig_h, orig_w) = img0_np.shape[:2]
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(img0_np)
    points = np.mgrid[:orig_h:32, :orig_w:32].reshape((2, -1))
    color = np.random.uniform(0.0, 1.0, size=(points.shape[1], 3))
    ax0.scatter(points[1], points[0], c=color)

    ax1.imshow(img1_np)
    points = (
        np.mgrid[:orig_h:32, :orig_w:32]
        + final_flow.squeeze(0).numpy(force=True)[:, ::32, ::32]
    ).reshape((2, -1))
    ax1.scatter(points[1], points[0], c=color)

    fig.show()
    breakpoint()


def visualize_optical_flow(flow_tensor: torch.Tensor) -> None:
    """
    Visualizes a (1, 2, H, W) optical flow map using Matplotlib.

    Args:
    - flow_tensor (torch.Tensor): A tensor of shape (1, 2, H, W) representing the optical flow.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import hsv_to_rgb

    # Ensure the tensor is on the CPU and in numpy format
    flow_tensor = flow_tensor.cpu().detach().numpy()

    # Extract the horizontal and vertical components of the flow
    flow_u = flow_tensor[0, 0, :, :]
    flow_v = flow_tensor[0, 1, :, :]

    # Compute the magnitude and angle of the flow
    magnitude = np.sqrt(flow_u**2 + flow_v**2)
    angle = np.arctan2(flow_v, flow_u)

    # Normalize magnitude to [0, 1]
    magnitude = magnitude / np.max(magnitude)

    # Create HSV image: Hue represents direction, Saturation represents magnitude
    hsv_image = np.zeros((flow_u.shape[0], flow_u.shape[1], 3), dtype=np.float32)
    hsv_image[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue: [0, 1] normalized angle
    hsv_image[..., 1] = magnitude  # Saturation: normalized magnitude
    hsv_image[..., 2] = 1  # Value: full brightness

    # Convert HSV image to RGB
    rgb_image = hsv_to_rgb(hsv_image)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title("Optical Flow Visualization")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    tyro.cli(run_optical_flow)
