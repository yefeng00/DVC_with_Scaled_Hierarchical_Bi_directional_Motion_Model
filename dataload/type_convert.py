import torch

def color_conv_matrix(type = "709"):
    if type=="601":
        # BT.601
        a = 0.299
        b = 0.587
        c = 0.114
        d = 1.772
        e = 1.402
    elif type=="709":
        # BT.709
        a = 0.2126
        b = 0.7152
        c = 0.0722
        d = 1.8556
        e = 1.5748
    elif type=="2020":
        # BT.2020
        a = 0.2627
        b = 0.6780
        c = 0.0593
        d = 1.8814
        e = 1.4747
    else:
        raise NotImplementedError

    return a,b,c,d,e

def yuv_to_rgb(image: torch.Tensor, type="709") -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5

    Took from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html#rgb_to_yuv
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                        .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :] - 0.5
    v: torch.Tensor = image[..., 2, :, :] - 0.5

    #r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    #g: torch.Tensor = y + -0.396 * u - 0.581 * v
    #b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    a,b,c,d,e = color_conv_matrix(type)

    
    r: torch.Tensor = y + e * v  # coefficient for g is 0
    g: torch.Tensor = y - (c * d / b) * u - (a * e / b) * v
    b: torch.Tensor = y + d * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out


def rgb_to_yuv(image: torch.Tensor, type="709") -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    a1,b1,c1,d1,e1 = color_conv_matrix(type)

    #y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    #u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    #v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b
    y: torch.Tensor = a1 * r + b1 * g + c1 * b
    u: torch.Tensor = (b - y) / d1 + 0.5
    v: torch.Tensor = (r - y) / e1 + 0.5

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out        
    