
import numpy as np
import pathlib
from typing import Callable, Dict, Literal, Optional, Protocol, Sequence, Tuple, Union


# where "ArrayLike" is very roughly ...
class ArrayLike(Protocol):
    shape: Tuple[int, ...]
    ndim: int
    dtype: np.dtype

    def __array__(self) -> np.ndarray: pass
    def __getitem__(self, key) -> ArrayLike: pass

LayerTypeName = Literal[
    'image', 'labels', 'points', 'shapes', 'surface', 'tracks', 'vectors'
]
LayerProps = Dict
DataType = Union[ArrayLike, Sequence[ArrayLike]]
FullLayerData = Tuple[DataType, LayerProps, LayerTypeName]
LayerData = Union[Tuple[DataType], Tuple[DataType, LayerProps], FullLayerData]

PathLike = str | pathlib.Path
PathOrPaths = PathLike | Sequence[PathLike]
ReaderFunction = Callable[[PathOrPaths], list[LayerData]]