# Napari Organoid Analyzer


A napari plugin to automatically detect, segment and analyze organoids from microscopy imaging data.


## Installation

This plugin has been tested with python 3.10 - you may consider using conda or pyenv to create your dedicated environment before running the `napari-organoid-analyzer`.

1. Install `napari-organoid-analyzer` 

    ```pip install git+https://github.com/Meleray/napari-organoid-analyzer@intel-mac```

2. Install `segment-anything`

```pip install git+https://github.com/facebookresearch/segment-anything.git```

Or 
```git clone https://github.com/facebookresearch/segment-anything.git```
```cd segment-anything```
```pip install .```

3. Manually install OpenMMLab dependencies:

     ``` 
    mim install mmengine
    mim install "mmcv==2.1.0"
    mim install mmdet
     ```

## How to use?
After installing, you can start napari (either by typing ```napari``` in your terminal or by launching the application) and select the plugin from the drop down menu.

## Contributing

Contributions are very welcome. Tests can be run with [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-organoid-analyzer" is free and open source software

## Dependencies


```napari-organoid-analyzer``` uses the ```napari-aicsimageio```<sup>[1]</sup> <sup>[2]</sup> plugin for reading and processing CZI images.

## Acknowledgements

The ```napari-organoid-analyzer``` is an extension of ```napari-organoid-counter``` plugin <sup>[4]</sup> ([Github](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter)). SAM-based organoid detection and segmentation are is implemented from [SAM_with_Detection_Head](https://github.com/Hanyi11/SAM_with_Detection_Head) by Hanyi Zhang and Lion Gleiter.

## References

[1] Eva Maxfield Brown, Dan Toloudis, Jamie Sherman, Madison Swain-Bowden, Talley Lambert, AICSImageIO Contributors (2021). AICSImageIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/AllenCellModeling/aicsimageio

[2] Eva Maxfield Brown, Talley Lambert, Peter Sobolewski, Napari-AICSImageIO Contributors (2021). Napari-AICSImageIO: Image Reading in Napari using AICSImageIO [Computer software]. GitHub. https://github.com/AllenCellModeling/napari-aicsimageio

The latest version also uses models developed with the ```mmdetection``` package <sup>[3]</sup>, see [here](https://github.com/open-mmlab/mmdetection)

[3] Chen, Kai, et al. "MMDetection: Open mmlab detection toolbox and benchmark." arXiv preprint arXiv:1906.07155 (2019).

[4] Christina Bukas, Harshavardhan Subramanian, & Marie Piraud. (2023). HelmholtzAI-Consultants-Munich/napari-organoid-counter: v0.2.0 (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.7859571