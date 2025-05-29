from qtpy.QtWidgets import QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QCheckBox, QLineEdit, QFileDialog
from qtpy.QtCore import Qt
from napari_organoid_analyzer import settings

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
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Export to folder:"))
        self.path_input = QLineEdit()
        self.path_input.setText(parent.session_vars['export_folder'])
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
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", self.parent().session_vars['export_folder'])
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
