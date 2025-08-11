from qtpy.QtWidgets import QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QCheckBox, QLineEdit, QFileDialog, QComboBox, QStackedLayout, QMessageBox
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
    def __init__(self, parent, available_features, masks_avilable, layer_ids):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setMinimumWidth(500)
        self.masks_available = masks_avilable
        self.layer_ids = layer_ids
        
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
        warning_label = QLabel("WARNING: It is recommended to export full instance masks only for small subset of detections, as they take up a lot of space. For exporting them as polygons use layer data export option.")
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)
        layout.addWidget(QLabel("Select what:"))
        
        # Export options layout (left side of the bottom part)
        options_layout = QVBoxLayout()
        
        # Checkboxes for export options
        self.export_layer_data = QCheckBox("Layer data (JSON)")
        self.export_layer_data.setChecked(True)
        self.export_instance_masks = QCheckBox("Instance masks (NPY)")
        self.export_instance_masks.setChecked(True)
        self.export_collated_mask = QCheckBox("Collated mask (NPY))")
        self.export_collated_mask.setChecked(True)
        self.export_features = QCheckBox("Features (CSV)")
        self.export_features.setChecked(True)
        self.export_features.stateChanged.connect(self._toggle_feature_selection)
        
        options_layout.addWidget(self.export_layer_data)
        options_layout.addWidget(self.export_features)
        if self.masks_available:
            options_layout.addWidget(self.export_instance_masks)
            # Selected IDs input for instance masks
            ids_layout = QHBoxLayout()
            ids_layout.addWidget(QLabel("Selected IDs:"))
            self.selected_ids_input = QLineEdit()
            self.selected_ids_input.setPlaceholderText("e.g., 1,2,5-10 (empty = none)")
            ids_layout.addWidget(self.selected_ids_input)
            options_layout.addLayout(ids_layout)
            
            # Connect checkbox state to enable/disable the input
            self.export_instance_masks.stateChanged.connect(self._toggle_ids_input)
            
            options_layout.addWidget(self.export_collated_mask)
        
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
        
        feature_layout.addStretch()
        self.feature_selection_widget.setLayout(feature_layout)
        self.feature_selection_widget.setVisible(True)
        bottom_layout.addWidget(self.feature_selection_widget)
        layout.addLayout(bottom_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        export_button = QPushButton("Export")
        export_button.clicked.connect(self._validate_and_accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(export_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", session.SESSION_VARS.get('export_folder', ""))
        if folder:
            self.path_input.setText(folder)
            session.set_session_var('export_folder', folder)
    
    def _toggle_feature_selection(self, state):
        self.feature_selection_widget.setVisible(state == Qt.Checked)
    
    def _toggle_ids_input(self, state):
        """Enable/disable the selected IDs input based on instance masks checkbox"""
        self.selected_ids_input.setEnabled(state == Qt.Checked)
    
    def _validate_ids(self):
        """Validate the selected IDs input and return parsed IDs or None if invalid"""
        ids_text = self.selected_ids_input.text().strip()
        if not ids_text:
            return []
        
        selected_ids = set()
        for token in ids_text.split(','):
            token = token.strip()
            if token == "":
                QMessageBox.warning(self, "Invalid IDs", "Empty token encountered in ID list.")
                return None
            
            if '-' in token:
                range_data = token.split('-')
                if len(range_data) != 2:
                    QMessageBox.warning(self, "Invalid IDs", f"Invalid range format: {token}")
                    return None
                try:
                    start = int(range_data[0])
                    end = int(range_data[1])
                    if start < 0 or end < 0 or start > end:
                        QMessageBox.warning(self, "Invalid IDs", f"Invalid range values: {token}")
                        return None
                    for curr_id in range(start, end + 1):
                        if curr_id not in self.layer_ids:
                            QMessageBox.warning(self, "Invalid IDs", f"ID {curr_id} not found in layer.")
                            return None
                        selected_ids.add(curr_id)
                except ValueError:
                    QMessageBox.warning(self, "Invalid IDs", f"Invalid range format: {token}")
                    return None
            else:
                try:
                    curr_id = int(token)
                    if curr_id not in self.layer_ids:
                        QMessageBox.warning(self, "Invalid IDs", f"ID {curr_id} not found in layer.")
                        return None
                    selected_ids.add(curr_id)
                except ValueError:
                    QMessageBox.warning(self, "Invalid IDs", f"Invalid ID: {token}")
                    return None
        
        return list(map(int, selected_ids))
    
    def _validate_and_accept(self):
        """Validate inputs before accepting"""
        if self.masks_available and self.export_instance_masks.isChecked():
            parsed_ids = self._validate_ids()
            if parsed_ids is None:
                return  # Validation failed, don't close dialog
            self.parsed_selected_ids = parsed_ids
        else:
            self.parsed_selected_ids = None
        
        self.accept()
    
    def get_export_path(self):
        return self.path_input.text()
    
    def get_export_options(self):
        options = {
            'layer_data': self.export_layer_data.isChecked(),
            'instance_masks': self.masks_available and self.export_instance_masks.isChecked(),
            'collated_mask': self.masks_available and self.export_collated_mask.isChecked(),
            'features': self.export_features.isChecked()
        }
        
        if options['instance_masks'] and hasattr(self, 'parsed_selected_ids'):
            options['selected_ids'] = self.parsed_selected_ids
        
        return options
    
    def get_selected_features(self):
        return [feature for feature, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]
