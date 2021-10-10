import numpy as np
from typing import List, Tuple
import cv2

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    for i in range((len(names))):
        cv2.imshow(names[i], images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.
    If the paths have directories, they must already exist.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for i in range((len(filenames))):
        cv2.imwrite(filenames[i], images[i])
    return

def scale_down(image: np.array) -> np.array:
    """Returns an image half the size of the original.

    Args:
        image: A numpy array with an opencv image

    Returns:
        A numpy array with an opencv image half the size of the original image
    """
    half = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return half

def separate_channels(colored_image: np.array) -> t_image_triplet:
    """Takes an BGR color image and splits it three images.

    Args:
        colored_image: an numpy array sized [HxWxC] where the channels are in BGR (Blue, Green, Red) order

    Returns:
        A tuple with three BGR images the first one containing only the Blue channel active, the second one only the
        green, and the third one only the red.
    """
    blue = colored_image.copy()
    green = colored_image.copy()
    red = colored_image.copy()

    blue[:, :, 1] = 0
    blue[:, :, 2] = 0

    green[:, :, 0] = 0
    green[:, :, 2] = 0

    red[:, :, 0] = 0
    red[:, :, 1] = 0

    return blue, green, red
