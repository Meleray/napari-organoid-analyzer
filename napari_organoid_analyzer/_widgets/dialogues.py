from qtpy.QtWidgets import QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QCheckBox, QLineEdit, QFileDialog, QComboBox, QStackedLayout
from qtpy.QtCore import Qt
from napari_organoid_analyzer import settings
from datetime import datetime
from napari_organoid_analyzer import session


class ConfirmUpload(QDialog):
    '''
    The QDialog box that appears when the user selects to run organoid counter
    without having the selected model locally
    Parameters
    ----------
        parent: QWidget
            The parent widget, in this case an instance of OrganoidCounterWidget

    '''
    def __init__(self, parent: QWidget, model_name: str):
        super().__init__(parent)

        self.setWindowTitle("Confirm Download")
        # setup buttons and text to be displayed
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        text = (f"Model {model_name} not found locally. Downloading default model to \n"
                +str(settings.MODELS_DIR)+"\n"
                "This will only happen once. Click ok to continue or \n"
                "cancel if you do not agree. You won't be able to run\n"
                "the organoid counter if you click cancel.")
        # add all to layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel(text))
        hbox = QHBoxLayout()
        hbox.addWidget(ok_btn)
        hbox.addWidget(cancel_btn)
        layout.addLayout(hbox)
        self.setLayout(layout)
        # connect ok and cancel buttons with accept and reject signals
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

class ConfirmSamUpload(ConfirmUpload):
    '''
    The QDialog box that appears when the user selects to run organoid counter
    without having the SAM detection and segmentation model locally
    Parameters
    ----------
        parent: QWidget
            The parent widget, in this case an instance of OrganoidCounterWidget

    '''
    def __init__(self, parent: QWidget):
        super().__init__(parent, model_name="")
        text = ("SAM model not found locally. Downloading default model to \n"
                +str(settings.MODELS_DIR)+"\n"
                "This will only happen once. Click ok to continue or \n"
                "cancel if you do not agree. You won't be able to run\n"
                "the organoid segmentation and detection with SAMOS\n" 
                "if you click cancel. WARNING: The model size is 1.2 GB!")
        self.layout().itemAt(0).widget().setText(text)


class SignalDialog(QDialog):
    """
    Dialog for adding or selecting signal for image
    """
    def __init__(self, parent, image_layer_names):
        super().__init__(parent)
        self.setWindowTitle("Add signal layer")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        stacked_layout = QStackedLayout()

        signal_image_layout = QHBoxLayout()
        signal_image_layout.addWidget(QLabel("Selected signal layer:"), 2)
        self.signal_image_selector = QComboBox()
        self.signal_image_selector.addItems(image_layer_names)
        signal_image_layout.addWidget(self.signal_image_selector, 4)
        signal_image_widget = QWidget()
        signal_image_widget.setLayout(signal_image_layout)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Selected signal file:"), 2)
        self.path_input = QLineEdit()
        path_layout.addWidget(self.path_input, 3)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_file)
        path_layout.addWidget(browse_button, 1)
        path_widget = QWidget()
        path_widget.setLayout(path_layout)

        stacked_layout.addWidget(signal_image_widget)
        stacked_layout.addWidget(path_widget)

        signal_target_layout = QHBoxLayout()
        signal_target_layout.addWidget(QLabel("Image layer:"), 2)
        self.image_layer_selector = QComboBox()
        self.image_layer_selector.addItems(image_layer_names)
        signal_target_layout.addWidget(self.image_layer_selector, 4)
        layout.addLayout(signal_target_layout)

        upload_type_layout = QHBoxLayout()
        upload_type_layout.addWidget(QLabel("Signal source:"), 2)
        self.upload_type_selector = QComboBox()
        self.upload_type_selector.addItems(['Select existing layer', 'Upload signal image'])
        self.upload_type_selector.currentIndexChanged.connect(stacked_layout.setCurrentIndex)
        self.upload_type_selector.setCurrentIndex(0)
        self.upload_type_selector.setCurrentText('Select existing layer')
        upload_type_layout.addWidget(self.upload_type_selector, 4)
        layout.addLayout(upload_type_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Signal name: "), 2)
        self.name_textbox = QLineEdit()
        self.name_textbox.setText(f"Unnamed_Signal_{datetime.strftime(datetime.now(), '%H_%M_%S')}")
        name_layout.addWidget(self.name_textbox, 4)
        layout.addLayout(name_layout)

        layout.addLayout(stacked_layout)

        button_layout = QHBoxLayout()
        export_button = QPushButton("Import")
        export_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(export_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Image file')
        if file_path:
            self.path_input.setText(file_path)

    def get_target(self):
        return self.image_layer_selector.currentText()
    
    def get_source(self):
        if self.upload_type_selector.currentIndex() == 0:
            return True, self.signal_image_selector.currentText()
        else:
            return False, self.path_input.text()
        
    def get_name(self):
        if len(self.name_textbox.text()) > 0:
            return self.name_textbox.text()
        else:
            return f"Unnamed_Signal_{datetime.strftime(datetime.now(), '%H_%M_%S')}"
        
class SignalChannelDialog(QDialog):
    """
    Dialog for selecting signal channel
    """
    def __init__(self, parent, channel_num, signal_name):
        super().__init__(parent)
        layout = QVBoxLayout()
        text = QLabel(f"Multiple channels detected in the signal {signal_name}. Please select exact channel idx, containing the signal. Note: For RGB, 0 - R, 1 - G, 2 - B")
        text.setWordWrap(True)
        layout.addWidget(text)
        self.channel_selector = QComboBox()
        for i in range(channel_num):
            self.channel_selector.addItem(str(i))
        layout.addWidget(self.channel_selector)
        button_layout = QHBoxLayout()
        export_button = QPushButton("Confirm")
        export_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(export_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_channel_idx(self):
        return int(self.channel_selector.currentText())


class ExportDialog(QDialog):
    """
    Dialog for selecting export options
    """
    def __init__(self, parent, available_features):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setMinimumWidth(500)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Export path selection
        # TODO: put it in separate class to avoid copy/paste
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Export to folder:"))
        self.path_input = QLineEdit()
        self.path_input.setText(session.SESSION_VARS.get('export_folder', ""))
        path_layout.addWidget(self.path_input, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_folder)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)
        
        # What to export
        layout.addWidget(QLabel("Select what to export:"))
        
        # Export options layout (left side of the bottom part)
        options_layout = QVBoxLayout()
        
        # Checkboxes for export options
        self.export_bboxes = QCheckBox("Bounding boxes (JSON)")
        self.export_bboxes.setChecked(True)
        self.export_instance_masks = QCheckBox("Instance masks (NPY)")
        self.export_instance_masks.setChecked(True)
        self.export_collated_mask = QCheckBox("Collated mask (NPY)")
        self.export_collated_mask.setChecked(True)
        self.export_features = QCheckBox("Features (CSV)")
        self.export_features.setChecked(True)
        self.export_features.stateChanged.connect(self._toggle_feature_selection)
        
        options_layout.addWidget(self.export_bboxes)
        options_layout.addWidget(self.export_instance_masks)
        options_layout.addWidget(self.export_collated_mask)
        options_layout.addWidget(self.export_features)
        
        # Bottom part with options on left and feature selection on right
        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(options_layout)
        
        # Feature selection (right side)
        self.feature_selection_widget = QWidget()
        feature_layout = QVBoxLayout()
        feature_layout.addWidget(QLabel("Select features to export:"))
        
        self.feature_checkboxes = {}
        for feature in available_features:
            checkbox = QCheckBox(feature)
            checkbox.setChecked(True)  # Default checked
            self.feature_checkboxes[feature] = checkbox
            feature_layout.addWidget(checkbox)
        
        checkbox_bbox = QCheckBox("Bounding box")
        checkbox_bbox.setChecked(True)
        self.feature_checkboxes['Bounding box'] = checkbox_bbox
        feature_layout.addWidget(checkbox_bbox)
        
        feature_layout.addStretch()
        self.feature_selection_widget.setLayout(feature_layout)
        self.feature_selection_widget.setVisible(True)
        bottom_layout.addWidget(self.feature_selection_widget)
        layout.addLayout(bottom_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        export_button = QPushButton("Export")
        export_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(export_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", session.SESSION_VARS['export_folder'])
        if folder:
            self.path_input.setText(folder)
            self.parent().set_session_var('export_folder', folder)
    
    def _toggle_feature_selection(self, state):
        self.feature_selection_widget.setVisible(state == Qt.Checked)
    
    def get_export_path(self):
        return self.path_input.text()
    
    def get_export_options(self):
        return {
            'bboxes': self.export_bboxes.isChecked(),
            'instance_masks': self.export_instance_masks.isChecked(),
            'collated_mask': self.export_collated_mask.isChecked(),
            'features': self.export_features.isChecked()
        }
    
    def get_selected_features(self):
        return [feature for feature, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]
