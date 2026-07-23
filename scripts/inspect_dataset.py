import os
import sys
import torch
import matplotlib.image as mplimg

from glob import glob
from pathlib import Path

BSD68_DIRECTORY: str = 'data/BSD68'
ROOT_DIRECTORY_NAME: str = "selfsup_denoising_library"

SUPPORTED_EXTENSIONS: tuple = ('png', 'jpg', 'jpeg')

def read_images(dir: Path) -> tuple:
    """
    Read the images of the supported extensions found at the input directory.

    Args:
        dir (Path): input directory

    Returns:
        list of sorted image paths, list of found extensions
    """
    assert os.path.isdir(dir), f"The given directory [{dir}] is not a directory!"
    
    # Find and sort all image paths    
    img_paths = []
    ext_found = []
    for ext in SUPPORTED_EXTENSIONS:
        mock_path = str(dir) + "/*." + ext
        found_paths = glob(mock_path)
        img_paths.extend(found_paths)
        if len(found_paths) != 0:
            ext_found.append(ext)
    img_paths.sort()
    
    return img_paths, ext_found


def load_image_to_tensor(img_path:str) -> torch.Tensor:
    """
    Given a path, load it to a pytorch tensor in grayscale.

    Args:
        img_path (str): input image path

    Returns:
        Torch tensor containing the loaded image
    """
    # x = torch.load(img_path, map_location=torch.device("cpu"), weights_only=True)
    img_np = mplimg.imread(img_path)
    x = torch.from_numpy(img_np) 
    return x


if __name__ == '__main__':
    # Given the root directory of BSD68 return an ordered list of image paths
    
    # Read where the script have been called from
    cwd = Path(os.getcwd())

    # Find the root path
    if cwd.parts[-1] != ROOT_DIRECTORY_NAME:
        print("The script has not been called from the root directory of the " \
        "project!")
        cwd_parts: tuple = cwd.parts
        assert ROOT_DIRECTORY_NAME in cwd_parts, "The script has been called "
        "from outside the project scope!"

        root_idx: int = cwd_parts.index(ROOT_DIRECTORY_NAME)
        root_path = cwd
        for i in range(len(cwd_parts) - root_idx - 1):
            root_path = root_path.parent
    else:
        root_path = cwd    
    print(f"The root path is: {root_path}")

    data_path: Path = root_path.joinpath(BSD68_DIRECTORY)
    print(f"The data path is: {data_path}")
    print()

    sorted_imgs, extensions = read_images(data_path)
    if sorted_imgs == 0:
        print(f"No images have been found in the {data_path} directory!")
        print("Exiting the program...")
        sys.exit()
    else:
        print(f"Number of found images: {len(sorted_imgs)}")
        print(f"Sorted list (first {min(len(sorted_imgs), 10)}): " \
              f"{sorted_imgs[:min(len(sorted_imgs), 10)]}")
        print(f"Found the following extensions: {extensions}")

    tmp_img = load_image_to_tensor(sorted_imgs[0])
    print(f"Type of the loaded image: {type(tmp_img)}")
    print(f"Shape of the loaded image: {tmp_img.shape}")
