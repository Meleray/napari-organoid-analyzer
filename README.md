# Napari Organoid Analyzer


A napari plugin to automatically detect, segment, annotate, and analyze the morphology of organoids from microscopy imaging data.


## Installation

You may consider using conda or venv to create your dedicated python environment before running the `napari-organoid-analyzer`, for example with
```
python -m venv napari_organoid_analyzer
source napari_organoid_analyzer/bin/activate
```

Then, install `napari-organoid-analyzer` with
```
python -m pip install git+https://github.com/Meleray/napari-organoid-analyzer[all]
```
or if you need to read CZI or MP4 files:
```
python -m pip install git+https://github.com/Meleray/napari-organoid-analyzer[all,czifile]
```
or
```
python -m pip install git+https://github.com/Meleray/napari-organoid-analyzer[all,mp4]
```

## How to use?
After installing, you can start napari (either by typing ```napari``` in your terminal or by launching the application) and select the plugin from the `Plugins` drop down menu.

## Contributing

Contributions are very welcome. Tests can be run with [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-organoid-analyzer" is free and open source software

## Acknowledgements

The ```napari-organoid-analyzer``` is an extension of the ```napari-organoid-counter``` plugin <sup>[1]</sup> ([Github](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter)). 
SAM-based organoid detection and segmentation are implemented from [SAM_with_Detection_Head](https://github.com/Hanyi11/SAM_with_Detection_Head) by Hanyi Zhang and Lion Gleiter.

## References

[1] Christina Bukas, Harshavardhan Subramanian, & Marie Piraud. (2023). HelmholtzAI-Consultants-Munich/napari-organoid-counter: v0.2.0 (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.7859571
