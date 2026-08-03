# This is an example definition of a architecture for a model used to calculate features from organoid images.

# ==============IMPORTANT========================
# Architectures should be defined in a separate folder, named after the architecture (e.g. KNN, MobileNerV3).
# The folder should contain file __init__.py with and define a list __all__ with the name of the main architecture class.
# (!) __all__ should contain the name of only one class.
# Local imports are permitted within the architecture folder.
# Architecture class should contain the following attributes and methods:
# - architecture_name: str - name of the architecture, displayed in the GUI
# - architecture_description: str - short description of the architecture, displayed in the GUI
# - train_data_type: str - type of data that will be passed to train() method
# - config_parameters: dict - configuration parameters expected by this architecture
# - default_config: dict - default configuration for the architecture
# - __init__(self, config) - constructor that initializes the architecture with the provided config
# - train(self, training_data, training_labels) - method to train the architecture with provided training data and labels
# - predict(self, data) - method to make predictions on new data
# - save_model(self, dirpath) - method to save the trained model into a directory
# - load_model(self, dirpath) - method to load a previously saved model from a directory
# ==============================================

class ExampleArchitecture:
    """
    Example architecture for calculating features from organoid images.
    
    This architecture serves as a template for creating custom architectures in the napari organoid analyzer plugin.
    It should be extended with specific methods to calculate desired features, but all me
    """

    # Architecture name that will be displayed in the GUI
    architecture_name = "Example Architecture"

    # Description of the architecture will be displayed in the GUI. Make it short and informative.
    architecture_description = "This is an example architecture for calculating features from organoid images."

    # Type of data that will be passed to train() method. It can be "features", "images" or "combined".
    # "features" - current features of detected organoid as a non-normalized(!) array of floats.
    # "images" - list of non-normalized cropped images of detected organoids (could have different sizes)
    # "combined" - both features and images as a tuple (features, images)
    train_data_type = "features"

    # Configuration parameters expected by this architecture.
    # Every parameter will be converted into an editable field in the training GUI.
    config_parameters = {
        "float_parameter": "float", # Float parameter, e.g. learning rate
        "int_parameter": "int", # Integer parameter, e.g. batch size
        "str_parameter": "str", # String parameter, e.g. name for trained architecture
        "bool_parameter": "bool", # Boolean parameter, e.g. whether to normalize features
        "selector_parameter": ["option_1", "option_2", "option_3"], # Selector parameter, will create a dropdown menu in the GUI with listed options.
    }

    # Default configuration for the architecture
    # This will be used to pre-fill the configuration fields in the GUI
    default_config = {
        "float_parameter": 0.01,
        "int_parameter": 42,
        "str_parameter": "example_architecture",
        "selector_parameter": "option_1",
        "bool_parameter": True,
    }


    # Initialized the architecture with the provided config.
    def __init__(self, config):
        pass

    # Method to train the architecture with provided training data and labels
    def train(self, training_data, training_labels):
        pass

    # Method to make predictions on new data.
    def predict(self, data):
        pass

    # Method to save the trained model into a directory.
    def save_model(self, dirpath):
        pass

    # Method to load a previously saved model from a directory. Loaded model should be ready for predictions.
    def load_model(self, dirpath):
        pass