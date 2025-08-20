import numpy as np
import pandas as pd
import shutil
import os
import json
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, 
    QPushButton, QLabel, QFormLayout, QGroupBox, QListWidget,
    QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent

from napari.utils.notifications import show_warning, show_error, show_info
from napari_organoid_analyzer._training.architecture_manager import ArchitectureManager
from napari_organoid_analyzer._training.training_thread import TrainingThread
from napari_organoid_analyzer import settings
from napari_organoid_analyzer import session
import json

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# Class for storing info about trained models
@dataclass
class ModelMetadata:
    name: str
    description: str
    arch_name: str
    arch_config: Dict[str, Any]
    used_features: List[str]
    target_feature: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(
            name=data['name'],
            description=data['description'],
            arch_name=data['arch_name'],
            arch_config=data['arch_config'],
            used_features=data['used_features'],
            target_feature=data['target_feature']
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'arch_name': self.arch_name,
            'arch_config': self.arch_config,
            'used_features': self.used_features,
            'target_feature': self.target_feature
        }
    

class TrainingWidget(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.main_widget = parent
        self.architecture_configs = {}
        self.selected_train_layers = []
        self.selected_prediction_layers = []
        self.selected_training_features = []
        self.training_data = None
        self.cur_trained_arch_instance = None
        self.cur_trained_model_metadata = None
        self.models_data = {}
        self.training_thread = None

        self.setup_ui()
        
        self.architectures_dir = settings.ARCHITECTURES_DIR
        self.models_dir = settings.TRAINED_MODELS_DIR
        self._initialize_architectures_cache() 

        self.architectures_manager = ArchitectureManager(self.architectures_dir)
        self.refresh_architectures()
        self.refresh_models()

        self.on_architecture_changed(self.architecture_selector.currentText())
        self.on_data_source_changed(self.data_source_selector.currentText())

    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Architecture selection
        arch_group = QGroupBox("Architecture Selection")
        arch_layout = QVBoxLayout()

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Architecture:"), 1)
        self.architecture_selector = QComboBox()
        self.architecture_selector.currentTextChanged.connect(self.on_architecture_changed)
        selector_layout.addWidget(self.architecture_selector, 4)
        arch_layout.addLayout(selector_layout)
        
        arch_buttons_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh Architectures")
        self.refresh_button.clicked.connect(self.refresh_architectures)
        arch_buttons_layout.addWidget(self.refresh_button)
        
        self.import_button = QPushButton("Import Custom Architecture")
        self.import_button.clicked.connect(self._import_custom_architecture)
        arch_buttons_layout.addWidget(self.import_button)
        
        self.adjust_config_button = QPushButton("Adjust Config")
        self.adjust_config_button.clicked.connect(self.open_config_dialog)
        self.adjust_config_button.setEnabled(False)
        arch_buttons_layout.addWidget(self.adjust_config_button)
        
        arch_layout.addLayout(arch_buttons_layout)
        
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"), 1)
        self.arch_description = QTextEdit()
        self.arch_description.setMaximumHeight(60)
        self.arch_description.setReadOnly(True)
        desc_layout.addWidget(self.arch_description, 4)
        arch_layout.addLayout(desc_layout)
        
        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group)
               
        # Training data selection
        data_group = QGroupBox("Training Data")
        data_layout = QVBoxLayout()
        
        data_source_layout = QHBoxLayout()
        data_source_layout.addWidget(QLabel("Data Source:"), 1)
        self.data_source_selector = QComboBox()
        self.data_source_selector.addItems(["CSV File", "Label Layers"])
        self.data_source_selector.currentTextChanged.connect(self.on_data_source_changed)
        data_source_layout.addWidget(self.data_source_selector, 4)
        data_layout.addLayout(data_source_layout)
        
        data_selection_layout = QHBoxLayout()
        data_selection_layout.addWidget(QLabel("Selection:"), 1)
        
        self.data_selection_edit = QLineEdit()
        self.data_selection_edit.setReadOnly(True)
        self.data_selection_edit.setPlaceholderText("No selection made")
        data_selection_layout.addWidget(self.data_selection_edit, 3)
        
        self.data_selection_button = QPushButton("Browse")
        self.data_selection_button.clicked.connect(self.on_data_selection_button_clicked)
        data_selection_layout.addWidget(self.data_selection_button, 1)
        
        data_layout.addLayout(data_selection_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training information
        model_group = QGroupBox("Training Information")
        model_layout = QVBoxLayout()
        
        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(QLabel("Model Name:"), 1)
        self.model_name_edit = QLineEdit("my_organoid_model")
        model_name_layout.addWidget(self.model_name_edit, 4)
        model_layout.addLayout(model_name_layout)
        
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"), 1)
        self.model_description_edit = QTextEdit()
        self.model_description_edit.setMaximumHeight(50)
        self.model_description_edit.setPlaceholderText("Description for this model...")
        desc_layout.addWidget(self.model_description_edit, 4)
        model_layout.addLayout(desc_layout)
        
        self.feature_selection_widget = QWidget()
        self.feature_selection_layout = QHBoxLayout(self.feature_selection_widget)
        self.feature_selection_layout.setContentsMargins(0, 0, 0, 0)
        self.feature_selection_layout.addWidget(QLabel("Training Features:"), 1)
        
        self.feature_selection_edit = QLineEdit()
        self.feature_selection_edit.setReadOnly(True)
        self.feature_selection_edit.setPlaceholderText("No features selected")
        self.feature_selection_layout.addWidget(self.feature_selection_edit, 3)
        
        self.feature_selection_button = QPushButton("Select Features")
        self.feature_selection_button.clicked.connect(self.select_training_features)
        self.feature_selection_layout.addWidget(self.feature_selection_button, 1)
        
        model_layout.addWidget(self.feature_selection_widget)
        
        # Target feature selection
        self.target_selection_widget = QWidget()
        self.target_selection_layout = QHBoxLayout(self.target_selection_widget)
        self.target_selection_layout.setContentsMargins(0, 0, 0, 0)
        self.target_selection_label = QLabel("Target Feature:")
        self.target_selection_layout.addWidget(self.target_selection_label, 1)

        self.target_feature_selector = QComboBox()
        self.target_feature_selector.setPlaceholderText("Select target feature")
        self.target_selection_layout.addWidget(self.target_feature_selector, 4)
        
        model_layout.addWidget(self.target_selection_widget)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training controls
        controls_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_button)
        
        self.stop_button = QPushButton("Stop Training (do not use!")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        
        layout.addLayout(controls_layout)
        
        # Training output text field
        console_group = QGroupBox("Training Output")
        console_layout = QVBoxLayout()
        
        self.training_output = QTextEdit()
        self.training_output.setMaximumHeight(150)
        self.training_output.setReadOnly(True)
        self.training_output.setPlaceholderText("Training output will appear here...")
        self.training_output.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555555;
            }
        """)
        console_layout.addWidget(self.training_output)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        # Trained models list
        models_group = QGroupBox("Trained Models")
        models_layout = QVBoxLayout()

        self.models_list = QListWidget()
        self.models_list.setMaximumHeight(150)
        self.models_list.setSelectionMode(QListWidget.SingleSelection)
        models_layout.addWidget(self.models_list)
        self.models_list.itemSelectionChanged.connect(self.on_model_selection_changed)

        models_controls = QHBoxLayout()
        self.load_model_button = QPushButton("Export")
        self.load_model_button.clicked.connect(self.export_selected_model)
        self.delete_model_button = QPushButton("Delete")
        self.delete_model_button.clicked.connect(self.delete_selected_model)
        self.import_model_button = QPushButton("Import")
        self.import_model_button.clicked.connect(self.import_model)
        self.show_info_button = QPushButton("Show info")
        self.show_info_button.clicked.connect(self.show_model_info)


        models_controls.addWidget(self.load_model_button)
        models_controls.addWidget(self.delete_model_button)
        models_controls.addWidget(self.import_model_button)
        models_controls.addWidget(self.show_info_button)
        models_layout.addLayout(models_controls)
        
        layer_selection_layout = QHBoxLayout()
        layer_selection_layout.addWidget(QLabel("Layers:"), 1)
        
        self.prediction_layers_edit = QLineEdit()
        self.prediction_layers_edit.setReadOnly(True)
        self.prediction_layers_edit.setPlaceholderText("No layers selected")
        layer_selection_layout.addWidget(self.prediction_layers_edit, 3)
        
        self.select_prediction_layers_button = QPushButton("Select Layers")
        self.select_prediction_layers_button.clicked.connect(self.select_prediction_layers)
        layer_selection_layout.addWidget(self.select_prediction_layers_button, 1)
        
        models_layout.addLayout(layer_selection_layout)
        
        field_name_layout = QHBoxLayout()
        field_name_layout.addWidget(QLabel("Prediction Field:"), 1)
        self.prediction_field_name_edit = QLineEdit("prediction")
        self.prediction_field_name_edit.setPlaceholderText("Name for prediction field")
        field_name_layout.addWidget(self.prediction_field_name_edit, 4)
        models_layout.addLayout(field_name_layout)
        
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.perform_prediction)
        self.predict_button.setEnabled(False)
        models_layout.addWidget(self.predict_button)

        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _initialize_architectures_cache(self):
        """Initialize architectures cache by copying from builtin architectures folder."""
        try:
            source_architectures_dir = Path(__file__).parent / "builtin_architectures"
            
            if source_architectures_dir.exists():
                for arch_dir in source_architectures_dir.iterdir():
                    if arch_dir.is_dir() and not arch_dir.name.startswith('.') and not arch_dir.name.startswith('_'):
                        dest_dir = settings.ARCHITECTURES_DIR / arch_dir.name
                        
                        if dest_dir.exists():
                            shutil.rmtree(dest_dir)
                        shutil.copytree(arch_dir, dest_dir)
                        print(f"Updated architecture '{arch_dir.name}' in cache")

        except Exception as e:
            print(f"Error initializing architectures cache: {e}")
    
    def load_architecture_configs(self, arch_infos={}):
        """Load cached architecture configurations from session."""
        self.architecture_configs = {name: arch_info.default_config for name, arch_info in arch_infos.items()}
        try:
            session.load_cached_settings()
            cached_architecture_configs = session.SESSION_VARS.get('architecture_configs', {})
            for arch_name, config in cached_architecture_configs.items():
                if arch_name not in arch_infos:
                    show_warning(f"Architecture '{arch_name}' not found in current architectures. Skipping config load.")
                    continue
                self.architecture_configs[arch_name] = config
        except Exception as e:
            print(f"Error loading architecture configs: {e}")
            self.architecture_configs = {}
    
    def save_architecture_config(self, arch_name, config):
        """Save configuration for a specific architecture."""
        self.architecture_configs[arch_name] = config
        session.set_session_var('architecture_configs', self.architecture_configs)
    
    def open_config_dialog(self):
        """Open the configuration dialog for the current architecture."""

        arch_name = self.architecture_selector.currentText()

        if not arch_name:
            show_warning("Please select an architecture first.")
            return
        
        arch_info = self.architectures_manager.get_architecture_info(arch_name)

        if arch_info is None:
            show_warning(f"Architecture '{arch_name}' not found.")
            return
        
        current_config = self.architecture_configs[arch_name]
        dialog = ModelConfigDialog(self, arch_info, current_config)
        dialog.exec_()
    
    def _import_custom_architecture(self):
        """Import a custom architecture from user-selected folder."""
        dialog = ArchitectureImportDialog(self, self.architectures_manager)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_architectures()

    
    def refresh_architectures(self):
        """Refresh the list of available architectures."""
        self.architecture_selector.clear()
        arch_infos = self.architectures_manager.discover_architectures()
        self.load_architecture_configs(arch_infos)

        for arch_name in arch_infos.keys():
            self.architecture_selector.addItem(arch_name)
    
    def on_architecture_changed(self, arch_name):
        """Handle architecture selection change."""
        if not arch_name:
            self.adjust_config_button.setEnabled(False)
            return
        arch_info = self.architectures_manager.get_architecture_info(arch_name)
        if arch_info is None:
            show_warning(f"Architecture '{arch_name}' not found.")
            self.adjust_config_button.setEnabled(False)
            self.arch_description.clear()
            return
        self.adjust_config_button.setEnabled(True)
        self.arch_description.setText(arch_info.description)

        if arch_info.train_data_type == "images":
            self.feature_selection_widget.setVisible(False)
        else:
            self.feature_selection_widget.setVisible(True)

        if arch_info.train_data_type == "features":
            self.data_source_selector.clear()
            self.data_source_selector.addItems(["CSV File", "Label Layers"])
        elif arch_info.train_data_type == "images" or arch_info.train_data_type == "combined":
            self.data_source_selector.clear()
            self.data_source_selector.addItems(["Label Layers"])
    
    def on_data_source_changed(self, source):
        """Handle data source selection change."""
        self.training_data = None
        if source == "CSV File":
            self.data_selection_button.setText("Browse")
        elif source == "Label Layers":
            self.data_selection_button.setText("Select Layers")
        
        self.data_selection_edit.clear()
        self.selected_train_layers.clear()
        self.selected_training_features.clear()
        self.feature_selection_edit.clear()
        self.target_feature_selector.clear()
    
    def on_data_selection_button_clicked(self):
        """Handle data selection button click."""
        if self.data_source_selector.currentText() == "CSV File":
            self._browse_csv_file()
        elif self.data_source_selector.currentText() == "Label Layers":
            self._select_label_layers()

    def _browse_csv_file(self):
        """Open file dialog to select CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select CSV File", 
            session.SESSION_VARS.get('last_csv_dir', ""),
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            show_warning("No file selected")
            return

        self.data_selection_edit.setText(file_path)
        session.set_session_var('last_csv_dir', str(Path(file_path).parent))

        try:
            training_data_df = pd.read_csv(file_path)
            if training_data_df.empty:
                show_warning("Selected CSV file is empty or invalid.")
            else:
                show_info(f"Loaded {len(training_data_df)} rows from CSV file.")
            self.training_data = training_data_df
            self._update_feature_options()
        except Exception as e:
            show_error(f"Error loading CSV file: {str(e)}")
            return
    
    def _select_label_layers(self):
        """Open dialog to select label layers."""
        if not hasattr(self.main_widget, 'viewer') or self.main_widget.viewer is None:
            show_error("No napari viewer available")
            return
        
        label_layers = self.main_widget.shape_layer_names

        if len(label_layers) == 0:
            show_warning("No label layers found in the viewer")
            return
        
        dialog = LabelLayerSelectionDialog(self, label_layers, self.selected_train_layers)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_train_layers = dialog.get_selected_layers()
            if not self.selected_train_layers:
                self.data_selection_edit.setText("No label layers selected")
                return
            else:
                self.data_selection_edit.setText(", ".join(self.selected_train_layers))

            training_data_df = self.main_widget._get_features_from_layers(self.selected_train_layers)
            if training_data_df.empty:
                show_warning("Selected label layers contain no data.")
            else:
                show_info(f"Loaded {len(training_data_df)} rows from selected label layers.")
            self.training_data = training_data_df
            self._update_feature_options()

    
    def _update_feature_options(self):
        """Update available features and target features based on current training data."""
        self.target_feature_selector.clear()
        
        if self.training_data is None or self.training_data.empty:
            return
        
        for column in self.training_data.columns:
            self.target_feature_selector.addItem(column)
    
    def select_training_features(self):
        """Open dialog to select training features from the current data."""
        if self.training_data is None or self.training_data.empty:
            show_warning("No training data available. Please select a data source first.")
            return
        
        available_features = list(self.training_data.columns)
        
        dialog = FeatureSelectionDialog(self, available_features, self.selected_training_features)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_training_features = dialog.get_selected_features()
            if not self.selected_training_features:
                self.feature_selection_edit.clear()
            else:
                self.feature_selection_edit.setText(", ".join(self.selected_training_features))


    def start_training(self):
        """Start the training process for the selected architecture and data."""
        if self.training_data is None or self.training_data.empty:
            show_warning("No training data available. Please select a data source first.")
            return
        
        if not self.selected_training_features:
            show_warning("No training features selected. Please select features first.")
            return
        
        if not self.target_feature_selector.currentText():
            show_warning("No target feature selected. Please select a target feature first.")
            return
        
        model_name = self.model_name_edit.text().strip()
        if not model_name:
            show_warning("Model name cannot be empty.")
            return
        
        if model_name in self.models_data:
            reply = QMessageBox.question(
                self, 
                "Model Exists", 
                f"Model '{model_name}' already exists. Continuing training will overwrite it. Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        try:
            X_train = self.training_data[self.selected_training_features].values
            y_train = self.training_data[self.target_feature_selector.currentText()].values
        except KeyError as e:
            show_error(f"Selected feature not found in training data: {str(e)}")
            return
        
        arch_name = self.architecture_selector.currentText()
        if not arch_name:
            show_warning("No architecture selected.")
            return

        print(f"Loading architecture: {arch_name} with config: {self.architecture_configs[arch_name]}")
        self.cur_trained_arch_instance = self.architectures_manager.load_architecture(
            arch_name, 
            self.architecture_configs[arch_name]
        )
        if self.cur_trained_arch_instance is None:
            show_error(f"Failed to load architecture '{arch_name}'.")
            return
        
        self.cur_trained_model_metadata = ModelMetadata(
            name=self.model_name_edit.text(),
            description=self.model_description_edit.toPlainText(),
            arch_name=arch_name,
            arch_config=self.architecture_configs[arch_name],
            used_features=self.selected_training_features,
            target_feature=self.target_feature_selector.currentText()
        )

        self.training_output.clear()
        
        self.training_thread = TrainingThread(self.cur_trained_arch_instance, X_train, y_train)
        self.training_thread.output_signal.connect(self.append_training_output)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.training_thread.start()

    def stop_training(self):
        """Stop the currently running training process."""
        if self.training_thread and self.training_thread.isRunning():
            self.append_training_output("\nStopping training...")
            self.training_thread.stop()
            self.training_thread.wait(2000)
            
            if self.training_thread.isRunning():
                self.training_thread.terminate()
                self.append_training_output("Training forcefully terminated.\n")
            else:
                self.append_training_output("Training stopped.\n")
        
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def append_training_output(self, text):
        """Append text to the training output console."""
        self.training_output.append(text.rstrip())
        scrollbar = self.training_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_training_finished(self, success, message):
        """Handle training completion."""
        if success:
            show_info(message)
            self.append_training_output(f"\n✓ {message}")
        else:
            show_error(message)
            self.append_training_output(f"\n✗ {message}")
        
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_thread = None

        if not success:
            self.cur_trained_arch_instance = None
            self.cur_trained_model_metadata = None
            return

        # Save the trained model
        if not self.cur_trained_arch_instance or not self.cur_trained_model_metadata:
            show_error("No trained model to save.")
            return
        
        model_dir = self.models_dir / self.cur_trained_model_metadata.name
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = model_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.cur_trained_model_metadata.to_dict(), f, indent=4)
        except Exception as e:
            shutil.rmtree(model_dir)
            show_error(f"Failed to save model metadata: {e}")
            return
        
        checkpoint_dir = model_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.cur_trained_arch_instance.save_model(checkpoint_dir)
        except Exception as e:
            shutil.rmtree(model_dir)
            show_error(f"Failed to save model checkpoint: {e}")
            return

        show_info(f"Model '{self.cur_trained_model_metadata.name}' saved successfully.")
        self.cur_trained_arch_instance = None
        self.cur_trained_model_metadata = None
        self.refresh_models()

    def refresh_models(self):
        """Refresh the list of trained models."""
        self.models_data = {}
        
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.models_list.clear()
            return
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                print(f"Metadata file not found for model in {model_dir}. Skipping.")
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                model_metadata = ModelMetadata.from_dict(metadata_dict)
                self.models_data[model_metadata.name] = model_metadata
                
            except Exception as e:
                print(f"Error loading model metadata from {metadata_file}: {e}")
                continue

            print(f"Loaded model '{model_metadata.name}' with architecture '{model_metadata.arch_name}'")
        
        self.models_list.clear()
        for model_name in sorted(self.models_data.keys()):
            model_metadata = self.models_data[model_name]
            self.models_list.addItem(model_name)

    def export_selected_model(self):
        """Export the currently selected model."""
        current_item = self.models_list.currentItem()
        if not current_item:
            show_warning("No model selected for export.")
            return

        model_name = current_item.text()

        if model_name not in self.models_data:
            show_error(f"Model '{model_name}' not found in models data.")
            return
        
        model_metadata = self.models_data[model_name]
        
        destination_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select Export Destination", 
            session.SESSION_VARS.get('last_export_dir', "")
        )
        
        if not destination_dir:
            show_warning("No destination directory selected")
            return
        
        session.set_session_var('last_export_dir', destination_dir)
        export_path = Path(destination_dir) / model_name
        
        try:
            if export_path.exists():
                reply = QMessageBox.question(
                    self, 
                    "Directory Exists", 
                    f"Directory '{export_path}' already exists. Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                shutil.rmtree(export_path)
            
            export_path.mkdir(parents=True)
            
            source_model_dir = self.models_dir / model_name
            if not source_model_dir.exists():
                show_error(f"Source model directory '{source_model_dir}' not found.")
                return
            shutil.copytree(source_model_dir, export_path, dirs_exist_ok=True)
            
            arch_name = model_metadata.arch_name
            arch_info = self.architectures_manager.get_architecture_info(arch_name)
            
            if arch_info and arch_info.path.exists():
                architecture_export_path = export_path / "architecture"
                shutil.copytree(arch_info.path, architecture_export_path)
            else:
                raise ValueError(f"Exported model's architecture '{arch_name}' not found or invalid.")
            
            show_info(f"Model '{model_name}' exported successfully to '{export_path}'")
            
        except Exception as e:
            show_error(f"Failed to export model: {str(e)}")
            shutil.rmtree(export_path, ignore_errors=True)

    def delete_selected_model(self):
        """Delete the currently selected model."""
        current_item = self.models_list.currentItem()
        if not current_item:
            show_warning("No model selected for deletion.")
            return

        model_name = current_item.text()

        if model_name not in self.models_data:
            show_error(f"Model '{model_name}' not found in models data.")
            return
        
        reply = QMessageBox.question(
            self, 
            "Delete Model", 
            f"Are you sure you want to delete model '{model_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        try:
            model_dir = self.models_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            self.refresh_models()
            show_info(f"Model '{model_name}' deleted successfully.")
        except Exception as e:
            show_error(f"Failed to delete model: {str(e)}")

    def import_model(self):
        """Import a trained model from a file."""
        dialog = ModelImportDialog(self, self.architectures_manager, self.models_dir, is_import_mode=True)
        dialog.exec_()

    def show_model_info(self):
        """Show information about the selected model."""
        current_item = self.models_list.currentItem()
        if not current_item:
            show_warning("No model selected.")
            return
        
        model_name = current_item.text()
        
        if model_name not in self.models_data:
            show_error(f"Model '{model_name}' not found in models data.")
            return
        
        model_metadata = self.models_data[model_name]
        
        arch_info = self.architectures_manager.get_architecture_info(model_metadata.arch_name)
        if arch_info is None:
            show_warning(f"Architecture '{model_metadata.arch_name}' not found. Showing available model information only.")
        
        dialog = ModelImportDialog(self, self.architectures_manager, self.models_dir, is_import_mode=False)
        dialog.set_model_data(model_metadata, arch_info)
        dialog.exec_()
    
    def on_model_selection_changed(self):
        """Handle model list selection change for prediction."""
        current_item = self.models_list.currentItem()
        if not current_item:
            self.predict_button.setEnabled(False)
            return
        self.predict_button.setEnabled(len(self.selected_prediction_layers) > 0)
    
    def select_prediction_layers(self):
        """Open dialog to select layers for prediction."""
        if not hasattr(self.main_widget, 'viewer') or self.main_widget.viewer is None:
            show_error("No napari viewer available")
            return
        
        label_layers = self.main_widget.shape_layer_names
        
        if len(label_layers) == 0:
            show_warning("No label layers found in the viewer")
            return
        
        dialog = LabelLayerSelectionDialog(self, label_layers, self.selected_prediction_layers)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_prediction_layers = dialog.get_selected_layers()
            if not self.selected_prediction_layers:
                self.prediction_layers_edit.setText("No layers selected")
                self.predict_button.setEnabled(False)
            else:
                self.prediction_layers_edit.setText(", ".join(self.selected_prediction_layers))
                # Enable predict button only if both model and layers are selected
                current_item = self.models_list.currentItem()
                self.predict_button.setEnabled(current_item is not None)
    
    def perform_prediction(self):
        """Perform prediction on selected layers."""
        current_item = self.models_list.currentItem()
        if not current_item:
            show_warning("No model selected for prediction.")
            return
        
        model_name = current_item.text()
        if model_name not in self.models_data:
            show_error(f"Model '{model_name}' not found in models data.")
            return
        
        if not self.selected_prediction_layers:
            show_warning("No layers selected for prediction.")
            return
        
        prediction_field_name = self.prediction_field_name_edit.text().strip()
        if not prediction_field_name:
            show_warning("Prediction field name cannot be empty.")
            return
        
        try:
            model_metadata = self.models_data[model_name]
            model_dir = self.models_dir / model_name
            checkpoint_dir = model_dir / "checkpoint"
            
            if not checkpoint_dir.exists():
                show_error(f"Model checkpoint not found for '{model_name}'.")
                return
            
            arch_instance = self.architectures_manager.load_architecture(
                model_metadata.arch_name, 
                model_metadata.arch_config
            )
            
            if arch_instance is None:
                show_error(f"Failed to load architecture '{model_metadata.arch_name}' for model '{model_name}'.")
                return
            
            arch_instance.load_model(checkpoint_dir)
            
            print(f"Model '{model_name}' loaded successfully for prediction.")
            
            pred_model_metadata = model_metadata
            pred_model = arch_instance
            
        except Exception as e:
            show_error(f"Failed to load model '{model_name}': {str(e)}")
            return
        
        layers_with_property = []
        for layer_name in self.selected_prediction_layers:
            layer_data = self.main_widget._get_features_from_layers([layer_name])
            if not layer_data.empty and prediction_field_name in layer_data.columns:
                layers_with_property.append(layer_name)
        
        if layers_with_property:
            reply = QMessageBox.question(
                self,
                "Property Exists",
                f"The following layers already have property '{prediction_field_name}':\n" +
                "\n".join(layers_with_property) +
                f"\n\nDo you want to overwrite the existing property?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        skipped_layers = []
        predictions = {}
        
        for layer_name in self.selected_prediction_layers:
            try:
                result, preds = self._predict_on_layer(layer_name, pred_model, pred_model_metadata)
                if result == "success":
                    predictions[layer_name] = preds
                elif result == "skipped":
                    skipped_layers.append(layer_name)
                elif result == "cancelled":
                    return
            except Exception as e:
                show_error(f"Error predicting on layer '{layer_name}': {str(e)}")
                continue

        for layer_name, preds in predictions.items():
            cur_properties = self.main_widget.viewer.layers[layer_name].properties.copy()
            cur_properties[prediction_field_name] = preds
            self.main_widget.viewer.layers[layer_name].properties = cur_properties

        if len(predictions) > 0:
            show_info(f"Predictions completed for {len(predictions)} layers: {', '.join(predictions.keys())}")
        if skipped_layers:
            show_warning(f"Skipped {len(skipped_layers)} layers: {', '.join(skipped_layers)}")
        

    def _predict_on_layer(self, layer_name, pred_model, pred_model_metadata):
        """Perform prediction on a single layer. """
        layer_data = self.main_widget._get_features_from_layers([layer_name])
        if layer_data.empty:
            show_warning(f"No data found in layer '{layer_name}'. Skipping.")
            return "skipped"
        
        required_features = pred_model_metadata.used_features
        missing_features = [feat for feat in required_features if feat not in layer_data.columns]

        # In case of missing features, prompt user to map them, skip layer or cancel prediction        
        if missing_features:
            dialog = FeatureMappingDialog(self, layer_name, required_features, layer_data.columns)
            result = dialog.exec_()
            
            if result == QDialog.Rejected:
                return "cancelled", None
            elif dialog.should_skip:
                return "skipped", None
            else: 
                feature_mapping = dialog.get_feature_mapping()
                prediction_data = []
                for required_feature in required_features:
                    mapped_feature = feature_mapping[required_feature]
                    prediction_data.append(layer_data[mapped_feature].values)
                
                prediction_data = np.column_stack(prediction_data)
        else:
            prediction_data = layer_data[required_features].values
        
        predictions = pred_model.predict(prediction_data)
        
        return "success", predictions



class ArchitectureImportDialog(QDialog):
    """Dialog for importing custom architectures from directories."""
    
    def __init__(self, parent, architecture_manager):
        super().__init__(parent)
        self.architecture_manager = architecture_manager
        self.current_arch_info = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Import Custom Architecture")
        self.setModal(True)
        self.resize(600, 650)
        
        layout = QVBoxLayout()
        
        # Directory selection
        dir_group = QGroupBox("Select Architecture Directory")
        dir_layout = QHBoxLayout()
        
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setPlaceholderText("Select directory containing custom architecture...")
        self.dir_path_edit.textChanged.connect(self.on_directory_changed)
        dir_layout.addWidget(self.dir_path_edit)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_button)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Architecture preview (initially hidden)
        self.preview_group = QGroupBox("Architecture Preview")
        preview_layout = QVBoxLayout()
        
        info_layout = QVBoxLayout()
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_label = QLabel()
        self.name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(self.name_label)
        name_layout.addStretch()
        info_layout.addLayout(name_layout)
        
        data_type_layout = QHBoxLayout()
        data_type_layout.addWidget(QLabel("Data Type:"))
        self.data_type_label = QLabel()
        data_type_layout.addWidget(self.data_type_label)
        data_type_layout.addStretch()
        info_layout.addLayout(data_type_layout)
        
        preview_layout.addLayout(info_layout)
        
        desc_label = QLabel("Description:")
        preview_layout.addWidget(desc_label)
        self.description_text = QTextEdit()
        self.description_text.setMaximumHeight(80)
        self.description_text.setReadOnly(True)
        preview_layout.addWidget(self.description_text)
        
        deps_label = QLabel("Dependencies:")
        preview_layout.addWidget(deps_label)
        self.dependencies_text = QTextEdit()
        self.dependencies_text.setMaximumHeight(60)
        self.dependencies_text.setReadOnly(True)
        preview_layout.addWidget(self.dependencies_text)
        
        params_label = QLabel("Configuration Parameters:")
        preview_layout.addWidget(params_label)
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(["Name", "Type", "Default Value"])
        self.params_table.horizontalHeader().setStretchLastSection(True)
        self.params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.params_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.params_table.setMaximumHeight(200)
        preview_layout.addWidget(self.params_table)
        
        self.preview_group.setLayout(preview_layout)
        self.preview_group.setVisible(False)
        layout.addWidget(self.preview_group)
        
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.import_architecture)
        self.import_button.setEnabled(False)
        button_layout.addWidget(self.import_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def browse_directory(self):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Architecture Directory", session.SESSION_VARS.get('custom_arch_import_dir', "")
        )
        if directory:
            self.dir_path_edit.setText(directory)
            session.set_session_var('custom_arch_import_dir', directory)

    
    def on_directory_changed(self, path):
        """Handle directory path change."""
        if not path.strip():
            self.hide_preview()
            return
        
        try:
            arch_dir = Path(path)
            if not arch_dir.exists() or not arch_dir.is_dir():
                self.show_error("Selected path is not a valid directory")
                return
            
            arch_info = self.architecture_manager._parse_architecture_dir(arch_dir)
            
            if arch_info is None:
                self.show_error("No valid architecture found in the selected directory. Please look at the standard output for more details on the error.")
                return
            
            self.current_arch_info = arch_info
            self.show_preview(arch_info)
            
        except Exception as e:
            self.show_error(f"Error parsing architecture: {str(e)}")
    
    def show_preview(self, arch_info):
        """Show architecture preview."""
        self.name_label.setText(arch_info.name)
        self.description_text.setText(arch_info.description)
        self.data_type_label.setText(arch_info.train_data_type)
        
        dependencies_text = "\n".join(arch_info.dependencies) if arch_info.dependencies else "None"
        self.dependencies_text.setText(dependencies_text)
        
        self.populate_parameters_table(arch_info)
        
        self.preview_group.setVisible(True)
        self.error_label.setVisible(False)
        self.import_button.setEnabled(True)
    
    def populate_parameters_table(self, arch_info):
        """Populate the parameters table with configuration parameters."""
        config_params = arch_info.config_parameters
        default_config = arch_info.default_config
        
        self.params_table.setRowCount(len(config_params))
        
        for row, (param_name, param_spec) in enumerate(config_params.items()):
            name_item = QTableWidgetItem(param_name)
            self.params_table.setItem(row, 0, name_item)
            
            if isinstance(param_spec, str):
                type_text = param_spec
            elif isinstance(param_spec, list):
                type_text = f"choice: {', '.join(map(str, param_spec))}"
            else:
                type_text = str(type(param_spec).__name__)
            
            type_item = QTableWidgetItem(type_text)
            self.params_table.setItem(row, 1, type_item)
            
            default_value = default_config.get(param_name, "N/A")
            default_item = QTableWidgetItem(str(default_value))
            self.params_table.setItem(row, 2, default_item)
        
        self.params_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.params_table.resizeRowsToContents()
    
    def hide_preview(self):
        """Hide architecture preview."""
        self.preview_group.setVisible(False)
        self.error_label.setVisible(False)
        self.import_button.setEnabled(False)
        self.current_arch_info = None
        self.params_table.setRowCount(0)
    
    def show_error(self, message):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.preview_group.setVisible(False)
        self.import_button.setEnabled(False)
        self.current_arch_info = None
        self.params_table.setRowCount(0)
    
    def import_architecture(self):
        """Import the selected architecture."""
        if not self.current_arch_info:
            show_error("No valid architecture selected")
            return
        
        arch_info = self.current_arch_info
        source_path = Path(self.dir_path_edit.text())
        
        existing_architectures = self.architecture_manager.discover_architectures()
        if arch_info.name in existing_architectures:
            reply = QMessageBox.question(
                self,
                'Architecture Exists',
                f"Architecture '{arch_info.name}' already exists.\n\n"
                f"Do you want to replace it?\n"
                f"WARNING: This may make models previously trained with this architecture unusable.\n"
                f"Alternatively, you can adjust architecture_name property inside your architecture class",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        try:
            target_dir_name = source_path.name
            target_dir = self.architecture_manager.architectures_dir / target_dir_name
            
            if target_dir.exists():
                existing_arch_info = self.architecture_manager._parse_architecture_dir(target_dir)

                # Adjust folder name to avoid conflicts
                if existing_arch_info and existing_arch_info.name != arch_info.name:
                    counter = 1
                    while (self.architecture_manager.architectures_dir / f"{target_dir_name}_{counter}").exists():
                        counter += 1
                    target_dir_name = f"{target_dir_name}_{counter}"
                    target_dir = self.architecture_manager.architectures_dir / target_dir_name
            
            if arch_info.name in existing_architectures:
                existing_arch_info = existing_architectures[arch_info.name]
                if existing_arch_info.path.exists():
                    shutil.rmtree(existing_arch_info.path)
                del self.architecture_manager.discovered_architectures[arch_info.name]
            
            shutil.copytree(source_path, target_dir)
            
            show_info(f"Successfully imported architecture '{arch_info.name}' to {target_dir}")
            self.accept()
            
        except Exception as e:
            show_error(f"Failed to import architecture: {str(e)}")


class ModelConfigDialog(QDialog):
    """Dialog for adjusting architecture configuration parameters."""
    
    def __init__(self, parent, arch_info, current_config):
        super().__init__(parent)
        self.parent_widget = parent
        self.arch_info = arch_info
        self.current_config = current_config
        self.config_widgets = {}
        
        self.setWindowTitle(f"Configure {arch_info.name}")
        self.setModal(True)
        self.resize(400, 500)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout()
        
        desc_label = QLabel(f"Architecture: {self.arch_info.name}")
        desc_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(desc_label)
        
        desc_text = QTextEdit()
        desc_text.setText(self.arch_info.description)
        desc_text.setMaximumHeight(80)
        desc_text.setReadOnly(True)
        layout.addWidget(desc_text)
        
        # Configuration parameters
        config_group = QGroupBox("Configuration Parameters")
        self.config_layout = QFormLayout()

        config_params = self.arch_info.config_parameters
        for param_name, param_spec in config_params.items():
            widget = self.create_config_widget(param_name, param_spec)
            if widget:
                self.config_layout.addRow(f"{param_name}:", widget)
                self.config_widgets[param_name] = widget
                self.set_config_value(param_name, self.current_config[param_name])
                self.connect_widget_change(widget)

        config_group.setLayout(self.config_layout)
        layout.addWidget(config_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_button = QPushButton("Reset to Default")
        reset_button.clicked.connect(lambda: self.set_config_widgets(self.arch_info.default_config))
        button_layout.addWidget(reset_button)
        
        finish_button = QPushButton("Finish")
        finish_button.clicked.connect(self.accept)
        button_layout.addWidget(finish_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Prevent Enter key from resetting to default configuration."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:

            focused_widget = self.focusWidget()
            if isinstance(focused_widget, (QLineEdit, QSpinBox, QDoubleSpinBox)):
                self.save_current_config()
                event.accept()
                return
        super().keyPressEvent(event)
    
    def create_config_widget(self, param_name, param_spec):
        """Create a configuration widget based on parameter specification."""
        if isinstance(param_spec, str):
            if param_spec == "float":
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setDecimals(6)
                return widget
            elif param_spec == "int":
                widget = QSpinBox()
                widget.setRange(int(-1e9), int(1e9))
                return widget
            elif param_spec == "str":
                widget = QLineEdit()
                return widget
            elif param_spec == "bool":
                widget = QCheckBox()
                return widget
        elif isinstance(param_spec, list):
            widget = QComboBox()
            for option in param_spec:
                widget.addItem(str(option))
            return widget
        return None
    
    def set_config_value(self, param_name, value):
        """Set the value of a configuration parameter."""
        if param_name in self.config_widgets:
            widget = self.config_widgets[param_name]
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
        else:
            raise ValueError(f"Parameter '{param_name}' not found in configuration widgets")
        
    def set_config_widgets(self, config):
        """Set configuration values in the widgets."""
        for param_name, value in config.items():
            self.set_config_value(param_name, value)

    
    def connect_widget_change(self, widget):
        """Connect widget change signals to save configuration."""
        if isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(self.save_current_config)
        elif isinstance(widget, QSpinBox):
            widget.valueChanged.connect(self.save_current_config)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self.save_current_config)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self.save_current_config)
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self.save_current_config)

    
    def save_current_config(self):
        """Save current configuration to parent widget."""
        self.current_config = self.get_current_config()
        self.parent_widget.save_architecture_config(self.arch_info.name, self.current_config)

    def get_current_config(self):
        """Get the current configuration from the widgets."""
        config = {}
        config_params = self.arch_info.config_parameters

        for param_name, param_spec in config_params.items():
            if param_name in self.config_widgets:
                widget = self.config_widgets[param_name]
                
                if isinstance(param_spec, str):
                    if param_spec == "float":
                        config[param_name] = widget.value()
                    elif param_spec == "int":
                        config[param_name] = widget.value()
                    elif param_spec == "str":
                        config[param_name] = widget.text()
                    elif param_spec == "bool":
                        config[param_name] = widget.isChecked()
                elif isinstance(param_spec, list):
                    config[param_name] = widget.currentText()
        return config


class LabelLayerSelectionDialog(QDialog):
    """Dialog for selecting label layers from the napari viewer."""
    
    def __init__(self, parent, label_layers, selected_layers=[]):
        super().__init__(parent)
        self.label_layers = label_layers
        self.selected_layers = selected_layers
        self.checkboxes = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Select Label Layers")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Label layer checkboxes
        instruction_label = QLabel("Select the label layers to use for training:")
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for layer_name in self.label_layers:
            checkbox = QCheckBox(layer_name)
            if any(selected_name == layer_name for selected_name in self.selected_layers):
                checkbox.setChecked(True)

            self.checkboxes[layer_name] = (checkbox, layer_name)
            scroll_layout.addWidget(checkbox)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_button)
        
        select_none_button = QPushButton("Select None")
        select_none_button.clicked.connect(self.select_none)
        button_layout.addWidget(select_none_button)
        
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def select_all(self):
        """Select all checkboxes."""
        for checkbox, _ in self.checkboxes.values():
            checkbox.setChecked(True)
    
    def select_none(self):
        """Deselect all checkboxes."""
        for checkbox, _ in self.checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_layers(self):
        """Get the list of selected label layers."""
        selected = []
        for checkbox, layer in self.checkboxes.values():
            if checkbox.isChecked():
                selected.append(layer)
        return selected


class FeatureSelectionDialog(QDialog):
    """Dialog for selecting features from available columns."""
    
    def __init__(self, parent, available_features, selected_features=[]):
        super().__init__(parent)
        self.available_features = available_features
        self.selected_features = selected_features
        self.checkboxes = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Select Training Features")
        self.setModal(True)
        self.resize(400, 350)
        
        layout = QVBoxLayout()
        
        # Feature checkboxes
        instruction_label = QLabel("Select the features to use for training:")
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for feature_name in self.available_features:
            checkbox = QCheckBox(feature_name)
            if feature_name in self.selected_features:
                checkbox.setChecked(True)
            
            self.checkboxes[feature_name] = checkbox
            scroll_layout.addWidget(checkbox)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_button)
        
        select_none_button = QPushButton("Select None")
        select_none_button.clicked.connect(self.select_none)
        button_layout.addWidget(select_none_button)
        
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def select_all(self):
        """Select all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
    
    def select_none(self):
        """Deselect all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_features(self):
        """Get the list of selected features."""
        selected = []
        for feature_name, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                selected.append(feature_name)
        return selected


class FeatureMappingDialog(QDialog):
    """Dialog for mapping model features to layer properties when they don't match."""
    
    def __init__(self, parent, layer_name, required_features, available_properties):
        super().__init__(parent)
        self.layer_name = layer_name
        self.required_features = required_features
        self.available_properties = list(available_properties)
        self.feature_mapping = {}
        self.should_skip = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle(f"Feature Mapping - {self.layer_name}")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout()
        
        instruction_label = QLabel(
            f"Some required features are missing in layer '{self.layer_name}'.\n"
            f"Please map the required features to available properties:"
        )
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)
        
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(2)
        self.mapping_table.setHorizontalHeaderLabels(["Required Feature", "Available Property"])
        self.mapping_table.setRowCount(len(self.required_features))
        
        for row, feature in enumerate(self.required_features):
            feature_item = QTableWidgetItem(feature)
            feature_item.setFlags(feature_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(row, 0, feature_item)
            
            property_combo = QComboBox()
            property_combo.addItems(self.available_properties)
            
            if feature in self.available_properties:
                property_combo.setCurrentText(feature)
                self.feature_mapping[feature] = feature
            
            self.mapping_table.setCellWidget(row, 1, property_combo)
            property_combo.currentTextChanged.connect(
                lambda text, f=feature: self._on_mapping_changed(f, text)
            )
        
        self.mapping_table.horizontalHeader().setStretchLastSection(True)
        self.mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self.mapping_table)
        
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_mapping)
        button_layout.addWidget(self.confirm_button)
        
        self.skip_button = QPushButton("Skip Layer")
        self.skip_button.clicked.connect(self.skip_layer)
        button_layout.addWidget(self.skip_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        self._update_status()
    
    def _on_mapping_changed(self, feature, property_name):
        """Handle feature mapping change."""
        self.feature_mapping[feature] = property_name
        self._update_status()
    
    def _update_status(self):
        """Update status message."""
        missing_mappings = [f for f in self.required_features if f not in self.feature_mapping or not self.feature_mapping[f]]
        
        if missing_mappings:
            self.status_label.setText(f"Missing mappings: {', '.join(missing_mappings)}")
            self.confirm_button.setEnabled(False)
            return
        
        mapped_properties = list(self.feature_mapping.values())
        duplicates = [prop for prop in mapped_properties if mapped_properties.count(prop) > 1]
        
        if duplicates:
            self.status_label.setText(f"Duplicate mappings detected: {', '.join(set(duplicates))}")
            self.confirm_button.setEnabled(False)
            return
        
        self.status_label.setText("✓ Mapping is valid")
        self.status_label.setStyleSheet("color: green;")
        self.confirm_button.setEnabled(True)
    
    def confirm_mapping(self):
        """Confirm the feature mapping."""
        self.accept()
    
    def skip_layer(self):
        """Skip this layer."""
        self.should_skip = True
        self.accept()
    
    def get_feature_mapping(self):
        """Get the current feature mapping."""
        return self.feature_mapping.copy()


class ModelImportDialog(QDialog):
    """Dialog for importing or viewing model information."""
    
    def __init__(self, parent, architecture_manager, models_dir, is_import_mode=True):
        super().__init__(parent)
        self.parent_widget = parent
        self.architecture_manager = architecture_manager
        self.models_dir = models_dir
        self.is_import_mode = is_import_mode
        self.current_model_metadata = None
        self.current_arch_info = None
        self.model_dir_path = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        if self.is_import_mode:
            self.setWindowTitle("Import Model")
        else:
            self.setWindowTitle("Model Information")
        self.setModal(True)
        self.resize(600, 800)
        
        layout = QVBoxLayout()
        
        if self.is_import_mode:
            # Directory selection (only for import mode)
            self.dir_group = QGroupBox("Select Model Directory")
            dir_layout = QHBoxLayout()
            
            self.dir_path_edit = QLineEdit()
            self.dir_path_edit.setPlaceholderText("Select directory containing exported model...")
            self.dir_path_edit.textChanged.connect(self.on_directory_changed)
            dir_layout.addWidget(self.dir_path_edit)
            
            browse_button = QPushButton("Browse...")
            browse_button.clicked.connect(self.browse_directory)
            dir_layout.addWidget(browse_button)
            
            self.dir_group.setLayout(dir_layout)
            layout.addWidget(self.dir_group)
        
        # Model preview
        self.preview_group = QGroupBox("Model Information")
        preview_layout = QVBoxLayout()
        
        model_info_layout = QVBoxLayout()
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Model Name:"))
        self.model_name_label = QLabel()
        self.model_name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(self.model_name_label)
        name_layout.addStretch()
        model_info_layout.addLayout(name_layout)
        
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("Architecture:"))
        self.model_arch_label = QLabel()
        arch_layout.addWidget(self.model_arch_label)
        arch_layout.addStretch()
        model_info_layout.addLayout(arch_layout)
        
        features_label = QLabel("Used Features:")
        model_info_layout.addWidget(features_label)
        self.used_features_label = QTextEdit()
        self.used_features_label.setMaximumHeight(50)
        self.used_features_label.setReadOnly(True)
        model_info_layout.addWidget(self.used_features_label)
        
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Feature:"))
        self.target_feature_label = QLabel()
        target_layout.addWidget(self.target_feature_label)
        target_layout.addStretch()
        model_info_layout.addLayout(target_layout)
        
        preview_layout.addLayout(model_info_layout)
        
        desc_label = QLabel("Model Description:")
        preview_layout.addWidget(desc_label)
        self.model_description_text = QTextEdit()
        self.model_description_text.setMaximumHeight(50)
        self.model_description_text.setReadOnly(True)
        preview_layout.addWidget(self.model_description_text)
        
        self.preview_group.setLayout(preview_layout)
        if not self.is_import_mode:
            self.preview_group.setVisible(True)
        else:
            self.preview_group.setVisible(False)
        layout.addWidget(self.preview_group)
        
        # Architecture information
        self.arch_group = QGroupBox("Architecture Information")
        arch_layout = QVBoxLayout()
        
        arch_info_layout = QVBoxLayout()
        
        arch_name_layout = QHBoxLayout()
        arch_name_layout.addWidget(QLabel("Architecture Name:"))
        self.arch_name_label = QLabel()
        self.arch_name_label.setStyleSheet("font-weight: bold;")
        arch_name_layout.addWidget(self.arch_name_label)
        arch_name_layout.addStretch()
        arch_info_layout.addLayout(arch_name_layout)
        
        data_type_layout = QHBoxLayout()
        data_type_layout.addWidget(QLabel("Data Type:"))
        self.data_type_label = QLabel()
        data_type_layout.addWidget(self.data_type_label)
        data_type_layout.addStretch()
        arch_info_layout.addLayout(data_type_layout)
        
        arch_layout.addLayout(arch_info_layout)
        
        arch_desc_label = QLabel("Description:")
        arch_layout.addWidget(arch_desc_label)
        self.arch_description_text = QTextEdit()
        self.arch_description_text.setMaximumHeight(50)
        self.arch_description_text.setReadOnly(True)
        arch_layout.addWidget(self.arch_description_text)
        
        deps_label = QLabel("Dependencies:")
        arch_layout.addWidget(deps_label)
        self.dependencies_text = QTextEdit()
        self.dependencies_text.setMaximumHeight(50)
        self.dependencies_text.setReadOnly(True)
        arch_layout.addWidget(self.dependencies_text)
        
        config_label = QLabel("Configuration Parameters:")
        arch_layout.addWidget(config_label)
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(3)
        self.config_table.setHorizontalHeaderLabels(["Name", "Type", "Used Value"])
        self.config_table.horizontalHeader().setStretchLastSection(True)
        self.config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.config_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.config_table.setMaximumHeight(250)
        self.config_table.setMinimumHeight(150)
        arch_layout.addWidget(self.config_table)
        
        self.arch_group.setLayout(arch_layout)
        if not self.is_import_mode:
            self.arch_group.setVisible(True)
        else:
            self.arch_group.setVisible(False)
        layout.addWidget(self.arch_group)
        
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        if self.is_import_mode:
            self.import_button = QPushButton("Import")
            self.import_button.clicked.connect(self.import_model)
            self.import_button.setEnabled(False)
            button_layout.addWidget(self.import_button)
        
        close_button = QPushButton("Close" if not self.is_import_mode else "Cancel")
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def browse_directory(self):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", session.SESSION_VARS.get('model_import_dir', "")
        )
        if directory:
            self.dir_path_edit.setText(directory)
            session.set_session_var('model_import_dir', directory)
    
    def on_directory_changed(self, path):
        """Handle directory path change."""
        if not path.strip():
            self.hide_preview()
            return
        
        try:
            model_dir = Path(path)
            if not model_dir.exists() or not model_dir.is_dir():
                self.show_error("Selected path is not a valid directory")
                return
            
            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                self.show_error("No metadata.json found in the selected directory")
                return
            
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            self.current_model_metadata = ModelMetadata.from_dict(metadata_dict)
            self.model_dir_path = model_dir
            
            # Check for architecture directory
            arch_dir = model_dir / "architecture"
            if arch_dir.exists():
                self.current_arch_info = self.architecture_manager._parse_architecture_dir(arch_dir)
                if self.current_arch_info is None:
                    self.show_error("Invalid architecture found in the selected directory")
                    return
            else:
                self.show_error("No architecture directory found in the selected directory")
                return
            
            self.show_preview(self.current_model_metadata, self.current_arch_info)
            
        except Exception as e:
            self.show_error(f"Error parsing model directory: {str(e)}")
    
    def show_preview(self, model_metadata, arch_info):
        """Show model and architecture preview."""
        # Model information
        self.model_name_label.setText(model_metadata.name)
        self.model_arch_label.setText(model_metadata.arch_name)
        self.used_features_label.setText(", ".join(model_metadata.used_features))
        self.target_feature_label.setText(model_metadata.target_feature)
        self.model_description_text.setText(model_metadata.description)
        
        # Architecture information
        self.arch_name_label.setText(arch_info.name)
        self.data_type_label.setText(arch_info.train_data_type)
        self.arch_description_text.setText(arch_info.description)
        
        dependencies_text = "\n".join(arch_info.dependencies) if arch_info.dependencies else "None"
        self.dependencies_text.setText(dependencies_text)
        
        # Populate configuration table with model's arch_config
        self.populate_config_table(arch_info, model_metadata.arch_config)
        
        self.preview_group.setVisible(True)
        self.arch_group.setVisible(True)
        self.error_label.setVisible(False)
        if self.is_import_mode:
            self.import_button.setEnabled(True)
    
    def hide_preview(self):
        """Hide model preview."""
        self.preview_group.setVisible(False)
        self.arch_group.setVisible(False)
        self.error_label.setVisible(False)
        if self.is_import_mode:
            self.import_button.setEnabled(False)
        self.current_model_metadata = None
        self.current_arch_info = None
        self.model_dir_path = None
        self.config_table.setRowCount(0)
    
    def show_error(self, message):
        """Show error message."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.preview_group.setVisible(False)
        self.arch_group.setVisible(False)
        if self.is_import_mode:
            self.import_button.setEnabled(False)
        self.current_model_metadata = None
        self.current_arch_info = None
        self.model_dir_path = None
        self.config_table.setRowCount(0)
    
    def populate_config_table(self, arch_info, model_arch_config):
        """Populate the configuration table with the model's architecture configuration."""
        config_params = arch_info.config_parameters
        
        self.config_table.setRowCount(len(config_params))
        
        for row, (param_name, param_spec) in enumerate(config_params.items()):
            name_item = QTableWidgetItem(param_name)
            self.config_table.setItem(row, 0, name_item)
            
            if isinstance(param_spec, str):
                type_text = param_spec
            elif isinstance(param_spec, list):
                type_text = f"choice: {', '.join(map(str, param_spec))}"
            else:
                type_text = str(type(param_spec).__name__)
            
            type_item = QTableWidgetItem(type_text)
            self.config_table.setItem(row, 1, type_item)
            
            used_value = model_arch_config[param_name]
            value_item = QTableWidgetItem(str(used_value))
            self.config_table.setItem(row, 2, value_item)
        
        self.config_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.config_table.resizeRowsToContents()
    
    def set_model_data(self, model_metadata, arch_info):
        """Set model data for info mode (when not importing)."""
        if not self.is_import_mode:
            self.current_model_metadata = model_metadata
            self.current_arch_info = arch_info
            self.show_preview(model_metadata, arch_info)
    
    def import_model(self):
        """Import the selected model."""
        if not self.current_model_metadata or not self.current_arch_info or not self.model_dir_path:
            show_error("No valid model selected")
            return
        
        try:
            # First, try to import the architecture
            existing_architectures = self.architecture_manager.discover_architectures()
            arch_info = self.current_arch_info
            
            if arch_info.name in existing_architectures:
                reply = QMessageBox.question(
                    self,
                    'Architecture Exists',
                    f"Architecture '{arch_info.name}' already exists.\n\n"
                    f"Do you want to replace it?\n"
                    f"WARNING: This may make models previously trained with this architecture unusable.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
            
            arch_source_dir = self.model_dir_path / "architecture"
            target_dir_name = arch_source_dir.name
            target_dir = self.architecture_manager.architectures_dir / target_dir_name
            
            # Handle name conflicts for directory
            if target_dir.exists():
                existing_arch_info = self.architecture_manager._parse_architecture_dir(target_dir)
                if existing_arch_info and existing_arch_info.name != arch_info.name:
                    counter = 1
                    while (self.architecture_manager.architectures_dir / f"{target_dir_name}_{counter}").exists():
                        counter += 1
                    target_dir_name = f"{target_dir_name}_{counter}"
                    target_dir = self.architecture_manager.architectures_dir / target_dir_name
            
            if arch_info.name in existing_architectures:
                existing_arch_info = existing_architectures[arch_info.name]
                if existing_arch_info.path.exists():
                    shutil.rmtree(existing_arch_info.path)
                del self.architecture_manager.discovered_architectures[arch_info.name]
            
            shutil.copytree(arch_source_dir, target_dir)
            
            model_metadata = self.current_model_metadata
            if model_metadata.name in self.parent_widget.models_data:
                reply = QMessageBox.question(
                    self,
                    "Model Exists",
                    f"Model '{model_metadata.name}' already exists. Do you want to replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            model_target_dir = self.parent_widget.models_dir / model_metadata.name
            if model_target_dir.exists():
                shutil.rmtree(model_target_dir)
            
            model_target_dir.mkdir(parents=True, exist_ok=True)
            
            for item in self.model_dir_path.iterdir():
                if item.name != "architecture":
                    if item.is_file():
                        shutil.copy2(item, model_target_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, model_target_dir / item.name)
            
            show_info(f"Model '{model_metadata.name}' and architecture '{arch_info.name}' imported successfully")
            self.parent_widget.refresh_architectures()
            self.parent_widget.refresh_models()
            self.accept()
            
        except Exception as e:
            show_error(f"Failed to import model: {str(e)}")
