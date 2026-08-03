from pathlib import Path
from typing import Optional, Sequence
from napari.utils.notifications import Notification
import numpy as np


from ._napari_types import PathOrPaths, ReaderFunction, LayerData


def get_czi_reader(path: "PathOrPaths") -> Optional["ReaderFunction"]:
    # If we recognize the format, we return the actual reader function
    if isinstance(path, list) or isinstance(path, tuple):
        path = path[0]

    path = Path(path)

    print('\n\n')
    print(path)
    print('\n\n')

    if path.name.endswith(".czi"):
        # CZI file
        try:
            import czifile
        except ModuleNotFoundError:
            # CZI dependency is not installed
            Notification("Please install napari-organoid-analyzer with [all,czifile] dependency to read CZI files.")
            return None

        return czi_file_reader
    
    # otherwise we return None.
    return None

def czi_image_to_numpy(img):
    """Converts a single CZI image to a numpy array.

    Returns:
        image
    """
    # print('img.sizes', img.sizes)


    T = img.sizes.get('T', 1)
    C = img.sizes.get('C', 1)
    Z = img.sizes.get('Z', 1)
    Y = img.sizes.get('Y', 1)
    X = img.sizes.get('X', 1)

    # Treat unknown keys as separate channels
    unknown_keys = [k for k in img.sizes.keys() if k not in ('T', 'C', 'Z', 'Y', 'X')]
    C_factor = 1
    for k in unknown_keys:
        C_factor *= img.sizes[k]

    # Load in TCZYX format.
    kwargs = {k: None for k in (['T',] + unknown_keys + ['C', 'Z']) if k in img.sizes.keys()}
    tczyx = img(**kwargs).asarray()
    assert tczyx.shape[-2:] == (Y, X), tczyx.shape

    tczyx = tczyx.reshape((T, C_factor, C, Z, Y, X))
    assert tczyx.shape == (T, C_factor, C, Z, Y, X), tczyx.shape

    assert img.coord_scales.get('Y', None) == img.coord_scales.get('X', None), img.coord_scales
    assert img.coord_units.get('Y', None) == img.coord_units.get('X', None), img.coord_units
    px_size = img.coord_scales.get('X', None)
    px_unit = img.coord_units.get('X', None)

    # print('img.channels', img.channels)
    for j in range(C_factor):
        for i, channel_name in zip(range(C), img.channels.keys(), strict=True):
            tzyx = tczyx[:, j, i]
            tyx = np.max(tzyx, axis=1)  # Max projection
            yield tyx, {"px_size": px_size, 
                        "px_unit": px_unit, 
                        "channel": f"{j}_{channel_name}" if (C_factor > 1) else f"{channel_name}"}


def load_czi_file(path: Path, load_all_scenes=True) -> np.ndarray:
    """Loads all or the first image from a CZI file, together with its metadata.

    Returns:
        List of tuples (image, metadata), where image has the format [T,Y,X] 
        and metadata is {"px_size": px_size, "px_unit": px_unit, "channel": channel_name, "scene": scene_id}.
    """
    from czifile import CziFile

    images = []
    with CziFile(path) as czi:
        if load_all_scenes:
            for i, img in enumerate(czi.scenes.values()):
                for image, metadata in czi_image_to_numpy(img):
                    metadata['scene'] = i
                    images.append((image, metadata))
        else:
            for image, metadata in czi_image_to_numpy(czi.scenes[0]):
                assert image.shape[0] == 1, "Image is expected to be a single image not a timelapse."
                metadata['scene'] = 0
                images.append((image, metadata))

    return images


def czi_file_reader(path: "PathOrPaths") -> list["LayerData"]:
    if isinstance(path, list) or isinstance(path, tuple):
        # Assume timelapse, one scene per file, each scene has the same channels and same metadata.
        _data = []
        for p in path:
            _data.append(load_czi_file(Path(p), load_all_scenes=False))
        num_channels = len(_data[0])

        layer_data = []
        for ch in range(num_channels):
            layer_kwargs = {
                "name": f"{Path(p).stem}_{_data[0][ch][1]['channel']}",
                "metadata": {"px_size": _data[0][ch][1]['px_size'],
                             "px_unit": _data[0][ch][1]['px_unit']},
                "scale": None,
            }
            image = np.concatenate([d[ch][0] for d in _data], axis=0)
            layer_data.append((
                image,
                layer_kwargs,
                "image"
            ))
        return layer_data
    else:
        path = Path(path)

        layer_data = []
        for image, metadata in load_czi_file(path, load_all_scenes=True):
            layer_kwargs = {
                "name": f"{path.stem}_{metadata['scene']}_{metadata['channel']}",
                "metadata": {"px_size": metadata['px_size'],
                             "px_unit": metadata['px_unit']},
                "scale": None,
            }
            layer_data.append((
                image,
                layer_kwargs,
                "image"
            ))

        return layer_data