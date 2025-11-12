from napari_organoid_analyzer import session
from napari_organoid_analyzer import settings
from qtpy.QtCore import Qt, QLocale, QPoint
from qtpy.QtWidgets import (
    QHeaderView, 
    QAbstractItemView,
    QTableWidgetItem,
    QWidget, 
    QVBoxLayout, 
    QDialog, 
    QHBoxLayout, 
    QLabel, 
    QComboBox,
    QSpinBox,
    QPushButton, 
    QLineEdit, 
    QTableWidget,
    QStackedWidget,
    QSizePolicy,
    QMessageBox,
    QScrollArea,
    QCheckBox
)
from qtpy.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QDoubleValidator,
)
import numpy as np
import math
import cv2
import copy

def get_annotation_dialogue(image, layer_data, layer_properties, annotation_data, parent):
    type = annotation_data['type']
    if type == "Text":
        return TextAnnotationDialogue(image, layer_data, layer_properties, annotation_data, parent)
    elif type == "Ruler":
        return RulerAnnotationDialogue(image, layer_data, layer_properties, annotation_data, parent)
    elif type == 'Objects / Boxes':
        return BboxAnnotationWidget(image, layer_data, layer_properties, annotation_data, parent)
    elif type == 'Classes':
        return ClassAnnotationDialogue(image, layer_data, layer_properties, annotation_data, parent)
    elif type == 'Number':
        return NumberAnnotationDialogue(image, layer_data, layer_properties, annotation_data, parent)
    else:
        raise RuntimeError(f"Unknown annotation type {type}!")
    
class AnnotationDialogue(QDialog):
    def __init__(self, image, layer_data, layer_properties, annotation_data, parent=None):
        super().__init__(parent)
        self.layer_data = layer_data
        self.layer_properties = layer_properties
        self.annotation_name = annotation_data['annotation_name']
        self.property_name = annotation_data['property_name']
        self.annotations = annotation_data.get('data', {})
        self.id_str = annotation_data.get('ids', "")
        self.default_value = annotation_data.get('default_value', "")
        self.padding = int(annotation_data.get('padding', 10))
        self.border_width = int(annotation_data.get('border_width', 2))
        self.current_idx = -1

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w, ch = image.shape
        bytes_per_line = ch * w
        self.image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        layout = QVBoxLayout()

        self.stacked_widget = QStackedWidget()
        self.setup_widget = self._setup_start_widget()
        self.stacked_widget.addWidget(self.setup_widget)
        
        self.annotation_widget = self._setup_annotation_widget()
        self.stacked_widget.addWidget(self.annotation_widget)
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
        self.setWindowTitle("Annotation Tool")

    def _setup_start_widget(self):
        pass

    def _setup_annotation_widget(self):
        pass
        
class TextAnnotationDialogue(AnnotationDialogue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_start_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc = QLabel("INFO: This tool is for annotating text properties on existing label layers. Before starting annotation please adjust the following parameters:\n"
                      "Selected ids: Comma-separated Organoid IDs for annotation (if empty will use all organoids from selected layer; can select range of ids with \'-\' e.g. 1-10)\n"
                      "Default annotation value: Default value set for the annotated property\n"
                      "Padding: Padding around each bounding box in organoid visualization\n"
                      "Bounding box width: Width of detection bounding box in organoid visualization"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Selected ids
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Selected organoid ids:"))
        self.selected_ids_edit = QLineEdit()
        hbox.addWidget(self.selected_ids_edit)
        self.selected_ids_edit.setText(self.id_str)
        layout.addLayout(hbox)
        
        # Default value field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Default Annotation Value:"))
        self.default_value_edit = QLineEdit()
        self.default_value_edit.setText(self.default_value)
        hbox.addWidget(self.default_value_edit)
        layout.addLayout(hbox)
        
        # Padding field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Padding (pixels):"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(self.padding)
        hbox.addWidget(self.padding_spin)
        layout.addLayout(hbox)

        #Border width for bbox visualization
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Bounding box width (pixels):"))
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(self.border_width)
        hbox.addWidget(self.border_width_spin)
        layout.addLayout(hbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(super().reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_annotation)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _setup_annotation_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Image and properties layout
        content_layout = QHBoxLayout()
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.image_label.setMinimumSize(300, 300)
        content_layout.addWidget(self.image_label, 2)
        
        # Properties table
        self.props_table = QTableWidget()
        self.props_table.setColumnCount(2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.props_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content_layout.addWidget(self.props_table, 1)
        
        layout.addLayout(content_layout, 3)

        self.progress_label = QLabel("Annotation progress: ")
        layout.addWidget(self.progress_label)  
        
        # Annotation field
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotation:"))
        self.annotation_edit = QLineEdit()
        self.annotation_edit.returnPressed.connect(self.show_next)
        annot_layout.addWidget(self.annotation_edit)
        
        layout.addLayout(annot_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.cancel_btn2 = QPushButton("Cancel")
        self.cancel_btn2.clicked.connect(self.reject)
        nav_layout.addWidget(self.cancel_btn2)

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.show_prev)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.accept)
        nav_layout.addWidget(self.finish_btn)
        
        layout.addLayout(nav_layout)
        
        return widget
    
    def start_annotation(self):
        """Initialize annotation process"""
        # Get parameters from setup screen
        self.default_value = self.default_value_edit.text()
        self.padding = self.padding_spin.value()
        self.border_width = self.border_width_spin.value()

        def get_error_dialog(text):
            return QMessageBox.warning(self, "Invalid IDs", f"Invalid selected IDs string ({text}).")

        if len(self.selected_ids_edit.text().strip()) > 0:
            self.annotated_ids = set()
            for token in self.selected_ids_edit.text().strip().split(','):
                token = token.strip()
                if token == "":
                    get_error_dialog(f"Empty token encountered")
                    return
                if '-' in token:
                    range_data = token.split('-')
                    if len(range_data) != 2:
                        get_error_dialog(f"Invalid range {token}")
                        return
                    try:
                        start = int(range_data[0])
                        end = int(range_data[1])
                        if start < 0 or end < 0 or start > end:
                            get_error_dialog(f"Invalid range {token}")
                            return
                        for curr_id in range(start, end+1):
                            if not curr_id in self.layer_properties['bbox_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['bbox_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['bbox_id'])
        
        self.annotations = {key: val for key, val in self.annotations.items() if int(key) in self.annotated_ids}
        for box_id in self.annotated_ids:
            if not str(box_id) in self.annotations:
                self.annotations[str(box_id)] = self.default_value

        self.annotated_ids = list(self.annotated_ids)
        
        # Switch to annotation screen
        self.stacked_widget.setCurrentIndex(1)
        
        # Initialize annotation state
        self.current_idx = 0
        self.update_display()
    
    def update_display(self):
        """Update displayed bounding box"""
        box_id = self.annotated_ids[self.current_idx]
        layer_data_id = np.where(self.layer_properties['bbox_id'] == box_id)[0][0]
        x1, y1, x2, y2 = self.layer_data[layer_data_id]
        
        h, w = self.image.height(), self.image.width()
        x1_disp = max(0, int(x1) - self.padding)
        y1_disp = max(0, int(y1) - self.padding)
        x2_disp = min(h, int(x2) + self.padding)
        y2_disp = min(w, int(y2) + self.padding)
        
        # Display image snippet
        snippet = self.image.copy(y1_disp, x1_disp, y2_disp - y1_disp, x2_disp - x1_disp)

        painter = QPainter(snippet)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(self.border_width)
        painter.setPen(pen)
        painter.drawRect(
            self.padding + self.border_width // 2,
            self.padding + self.border_width // 2,
            int(y2 - y1 - self.border_width),
            int(x2 - x1 - self.border_width),
        )
        painter.end()

        pixmap = QPixmap.fromImage(snippet)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        self.props_table.setRowCount(len(self.layer_properties))
        for i, (prop, values) in enumerate(self.layer_properties.items()):
            value = values[layer_data_id]
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.props_table.item(i, 0).setToolTip(str(prop))
            self.props_table.item(i, 1).setToolTip(str(value))    
            
        # Set current annotation
        current_annot = self.annotations.get(str(box_id), self.default_value)
        self.annotation_edit.setText(str(current_annot))
        self.annotation_edit.setFocus()
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)

        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
        if self.current_idx >= 0:
            box_id = self.annotated_ids[self.current_idx]
            annotation = self.annotation_edit.text()
            self.annotations[str(box_id)] = annotation
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'property_name': self.property_name,
            'type': "Text",
            'data': self.annotations,
            'ids': self.selected_ids_edit.text().strip(),
            'default_value': self.default_value,
            'padding': self.padding,
            'border_width': self.border_width
        }
        session.set_session_var('annotation_features', annotation_features)

    
    def show_prev(self):
        """Show previous bounding box"""
        self.save_annotation()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
        else:
            self.current_idx = -1
            self.stacked_widget.setCurrentIndex(0)
    
    def show_next(self):
        """Show next bounding box"""
        self.save_annotation()
        if self.current_idx < len(self.annotated_ids) - 1:
            self.current_idx += 1
            self.update_display()
    
    def get_annotations(self):
        """Return the updated annotations dictionary"""
        return self.annotations
    
    def accept(self):
        self.save_annotation()
        super().accept()

    def reject(self):
        self.save_annotation()
        super().reject()
    
class NumberAnnotationDialogue(AnnotationDialogue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_start_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc = QLabel("INFO: This tool is for annotating number properties on existing label layers. Before starting annotation please adjust the following parameters:\n"
                      "Selected ids: Comma-separated Organoid IDs for annotation (if empty will use all organoids from selected layer; can select range of ids with \'-\'e.g. 1-10)\n"
                      "Default annotation value: Default value set for the annotated property (must be float)\n"
                      "Padding: Padding around each bounding box in organoid visualization\n"
                      "Bounding box width: Width of detection bounding box in organoid visualization"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Selected organoid ids:"))
        self.selected_ids_edit = QLineEdit()
        hbox.addWidget(self.selected_ids_edit)
        self.selected_ids_edit.setText(self.id_str)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Default Annotation Value:"))
        self.default_value_edit = QLineEdit()
        validator = QDoubleValidator()
        validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.default_value_edit.setValidator(validator)
        self.default_value_edit.setText(str(self.default_value))
        hbox.addWidget(self.default_value_edit)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Padding (pixels):"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(self.padding)
        hbox.addWidget(self.padding_spin)
        layout.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Bounding box width (pixels):"))
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(self.border_width)
        hbox.addWidget(self.border_width_spin)
        layout.addLayout(hbox)
        
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(super().reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_annotation)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _setup_annotation_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        content_layout = QHBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.image_label.setMinimumSize(300, 300)
        content_layout.addWidget(self.image_label, 2)
        
        self.props_table = QTableWidget()
        self.props_table.setColumnCount(2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.props_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content_layout.addWidget(self.props_table, 1)
        
        layout.addLayout(content_layout, 3)

        self.progress_label = QLabel("Annotation progress: ")
        layout.addWidget(self.progress_label)
        
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotation:"))
        self.annotation_edit = QLineEdit()
        validator = QDoubleValidator()
        validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.annotation_edit.setValidator(validator)
        self.annotation_edit.returnPressed.connect(self.show_next)
        annot_layout.addWidget(self.annotation_edit)
        
        layout.addLayout(annot_layout)
        
        nav_layout = QHBoxLayout()

        self.cancel_btn2 = QPushButton("Cancel")
        self.cancel_btn2.clicked.connect(self.reject)
        nav_layout.addWidget(self.cancel_btn2)

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.show_prev)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.accept)
        nav_layout.addWidget(self.finish_btn)
        
        layout.addLayout(nav_layout)
        
        return widget
    
    def start_annotation(self):
        """Initialize annotation process"""
        try:
            self.default_value = float(self.default_value_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                               "Please enter a valid number for the default value")
            return
        self.padding = self.padding_spin.value()
        self.border_width = self.border_width_spin.value()

        def get_error_dialog(text):
            return QMessageBox.warning(self, "Invalid IDs", f"Invalid selected IDs string ({text}).")

        if len(self.selected_ids_edit.text().strip()) > 0:
            self.annotated_ids = set()
            for token in self.selected_ids_edit.text().strip().split(','):
                token = token.strip()
                if token == "":
                    get_error_dialog(f"Empty token encountered")
                    return
                if '-' in token:
                    range_data = token.split('-')
                    if len(range_data) != 2:
                        get_error_dialog(f"Invalid range {token}")
                        return
                    try:
                        start = int(range_data[0])
                        end = int(range_data[1])
                        if start < 0 or end < 0 or start > end:
                            get_error_dialog(f"Invalid range {token}")
                            return
                        for curr_id in range(start, end+1):
                            if not curr_id in self.layer_properties['bbox_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['bbox_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['bbox_id'])
        
        self.annotations = {key: float(val) for key, val in self.annotations.items() if int(key) in self.annotated_ids}
        for box_id in self.annotated_ids:
            if not str(box_id) in self.annotations:
                self.annotations[str(box_id)] = self.default_value

        self.annotated_ids = list(self.annotated_ids)
        
        self.stacked_widget.setCurrentIndex(1)
        
        self.current_idx = 0
        self.update_display()
    
    def update_display(self):
        """Update displayed bounding box"""
        box_id = self.annotated_ids[self.current_idx]
        layer_data_id = np.where(self.layer_properties['bbox_id'] == box_id)[0][0]
        x1, y1, x2, y2 = self.layer_data[layer_data_id]
        
        h, w = self.image.height(), self.image.width()
        x1_disp = max(0, int(x1) - self.padding)
        y1_disp = max(0, int(y1) - self.padding)
        x2_disp = min(h, int(x2) + self.padding)
        y2_disp = min(w, int(y2) + self.padding)
        
        snippet = self.image.copy(y1_disp, x1_disp, y2_disp - y1_disp, x2_disp - x1_disp)

        painter = QPainter(snippet)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(self.border_width)
        painter.setPen(pen)
        painter.drawRect(
            self.padding + self.border_width // 2,
            self.padding + self.border_width // 2,
            int(y2 - y1 - self.border_width),
            int(x2 - x1 - self.border_width),
        )
        painter.end()

        pixmap = QPixmap.fromImage(snippet)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        self.props_table.setRowCount(len(self.layer_properties))
        for i, (prop, values) in enumerate(self.layer_properties.items()):
            value = values[layer_data_id]
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.props_table.item(i, 0).setToolTip(str(prop))
            self.props_table.item(i, 1).setToolTip(str(value))    
            
        current_annot = self.annotations.get(str(box_id), self.default_value)
        self.annotation_edit.setText(str(current_annot))
        self.annotation_edit.setFocus()
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)
        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
        if self.current_idx >= 0:
            box_id = self.annotated_ids[self.current_idx]
            try:
                annotation = float(self.annotation_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", 
                               "Please enter a valid number for the annotation")
                return
            self.annotations[str(box_id)] = annotation
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'property_name': self.property_name,
            'type': "Number",
            'data': self.annotations,
            'ids': self.selected_ids_edit.text().strip(),
            'default_value': self.default_value,
            'padding': self.padding,
            'border_width': self.border_width
        }
        session.set_session_var('annotation_features', annotation_features)

    
    def show_prev(self):
        """Show previous bounding box"""
        self.save_annotation()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
        else:
            self.current_idx = -1
            self.stacked_widget.setCurrentIndex(0)
    
    def show_next(self):
        """Show next bounding box"""
        self.save_annotation()
        if self.current_idx < len(self.annotated_ids) - 1:
            self.current_idx += 1
            self.update_display()
    
    def get_annotations(self):
        """Return the updated annotations dictionary"""
        return self.annotations
    
    def accept(self):
        self.save_annotation()
        super().accept()

    def reject(self):
        self.save_annotation()
        super().reject()
    

def get_length(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return math.sqrt(dx*dx+dy*dy)

def get_poly_line_length(line):
    """Calculate the length of a line segment"""
    if len(line) < 2:
        return 0.0
    total_length = 0.0
    for i in range(1, len(line)):
        if not isinstance(line[i], (list, tuple)) or len(line[i]) != 2:
            raise ValueError("Line points must be lists or tuples of two elements (y, x).")
        x1, y1 = line[i-1]
        x2, y2 = line[i]
        total_length += get_length(x1, y1, x2, y2)
    return total_length

def get_poly_lines_data(lines):
    lengths = [get_poly_line_length(line) for line in lines]
    total_length = sum(lengths)
    average_length = total_length / len(lengths) if lengths else 0.0
    count = len(lines)
    return total_length, average_length, count

class RulerImageLabel(QLabel):
    """Custom QLabel for multi-segment lines annotation tool"""
    def __init__(self, dialogue_class, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.setMouseTracking(True) 
        self._pixmap = None
        self._original_pixmap = None
        self.curr_point = None
        self.drawing = False
        self.line_color = QColor(255, 0, 0)
        self.line_width = 2
        self.dialogue_class = dialogue_class
        self.lines = []
        self.cur_snippet_coords = None

    def set_snippet_coords(self, coords):
        """Set global coordinates of the current image snippet's top-left corner"""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ValueError("Snippet coordinates must be a list or tuple of two elements (y, x).")
        self.cur_snippet_coords = coords

    def setPixmap(self, pixmap):
        """Store the original pixmap and update the display"""
        self._original_pixmap = pixmap
        self._pixmap = pixmap.copy() if pixmap else None
        self.clear_image()
        super().setPixmap(self.scaled_pixmap())
        
    def scaled_pixmap(self):
        """Return scaled pixmap based on current label size"""
        if self._pixmap is None or self._pixmap.isNull():
            return QPixmap()
        pixmap_size = self._pixmap.size()
        pixmap_size.scale(self.size(), Qt.IgnoreAspectRatio)
        return self._pixmap.scaled(pixmap_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        
    def resizeEvent(self, event):
        """Handle resize events to update the pixmap"""
        if self._pixmap and not self._pixmap.isNull():
            super().setPixmap(self.scaled_pixmap())
        super().resizeEvent(event)
        
    def mousePressEvent(self, event):
        """Line drawing controls"""
        if event.button() == Qt.LeftButton:
            self.cur_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            if not self.drawing:
                self.drawing = True
                x = self.cur_point.x()
                y = self.cur_point.y()
                self.lines.append([[x, y], [x, y]])
            else:
                x = self.cur_point.x()
                y = self.cur_point.y()
                self.lines[-1][-1] = [x, y]
                self.lines[-1].append([x, y])
        elif event.button() == Qt.RightButton and self.drawing:
            self.lines[-1].pop()
            if len(self.lines[-1]) == 1:
                self.lines.pop()
            self.drawing = False
        self.update_lines()
 
    def mouseMoveEvent(self, event):
        """Update last line segment when moving cursor"""
        if self.drawing:
            self.cur_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            x = self.cur_point.x()
            y = self.cur_point.y()
            self.lines[-1][-1] = [x, y]
            self.update_lines()

    def get_lines(self):
        if self.cur_snippet_coords is None:
            raise ValueError("Snippet coordinates must be set before getting bounding boxes.")
        global_lines = copy.deepcopy(self.lines)
        for line in global_lines:
            for point in line:
                point[0] += self.cur_snippet_coords[0]
                point[1] += self.cur_snippet_coords[1]
                tmp = point[0]
                point[0] = point[1]
                point[1] = tmp
        return global_lines
    
    def set_lines(self, lines):
        if self.cur_snippet_coords is None:
            raise ValueError("Snippet coordinates must be set before setting bounding boxes.")
        self.lines = []
        for line in lines:
            for point in line:
                tmp = point[0]
                point[0] = point[1]
                point[1] = tmp
                point[0] -= self.cur_snippet_coords[0]
                point[1] -= self.cur_snippet_coords[1]
            self.lines.append(line)
        self.update_lines()
            
    def clear_image(self):
        """Clear the current ruler line"""
        self.lines = []
        self.update_lines()

    def clear_last_segment(self):
        """Clear last drawn line segment"""
        if len(self.lines):
            if self.drawing:
                self.lines[-1] = self.lines[-1][:-2]
                self.lines[-1].append(self.lines[-1][-1])
                if len(self.lines[-1]) <= 2:
                    self.lines.pop()
                    self.drawing = False
            else:
                self.lines[-1] = self.lines[-1][:-1]
                if len(self.lines[-1]) <= 1:
                    self.lines.pop()
                    self.drawing = False
            self.update_lines()
            
    def update_lines(self):
        """Redraw lines on the image"""
        if self._original_pixmap:
            self._pixmap = self._original_pixmap.copy()
            painter = QPainter(self._pixmap)
            pen = QPen(self.line_color)
            pen.setWidth(self.line_width)
            painter.setPen(pen)

            for line in self.lines:
                painter.drawPolyline(
                    *[QPoint(point[0], point[1]) for point in line]
                )      
            painter.end()
            super().setPixmap(self.scaled_pixmap())
            global_lines = self.get_lines()
            total_length, average_length, count = get_poly_lines_data(global_lines)
            if len(global_lines):
                curr_point = global_lines[-1][-1]
                self.dialogue_class.coord_label.setText(f"Current coordinates: {curr_point[0]}, {curr_point[1]}")
            else:
                self.dialogue_class.coord_label.setText("Current coordinates: None")
            self.dialogue_class.annotation_edit.setText(str(global_lines))
            self.dialogue_class.annotation_total_length.setText(str(total_length))
            self.dialogue_class.annotation_average_length.setText(str(average_length))
            self.dialogue_class.annotation_count.setText(str(count))

    def original_to_scaled(self, point):
        """Convert a point from original image coordinates to label coordinates"""
        label_size = self.size()
        pixmap_size = self._original_pixmap.size()

        scale_factor_x = label_size.width() / pixmap_size.width()
        scale_factor_y = label_size.height() / pixmap_size.height()
        
        scaled_x = point.x() * scale_factor_x
        scaled_y = point.y() * scale_factor_y
        
        return QPoint(int(scaled_x), int(scaled_y))
            
    def scaled_to_original(self, point):
        """Convert a point from label coordinates to original image coordinates"""
        label_size = self.size()
        pixmap_size = self._original_pixmap.size()
        
        scale_factor_x = label_size.width() / pixmap_size.width()
        scale_factor_y = label_size.height() / pixmap_size.height()
        
        orig_x = point.x() / scale_factor_x
        orig_y = point.y() / scale_factor_y
        
        return QPoint(int(orig_x), int(orig_y))
        
class RulerAnnotationDialogue(AnnotationDialogue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_start_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc = QLabel("INFO: This tool is for annotating properties with ruler tool. You could draw a single line on every image. The feature will be populated with line length."
                      "Before starting annotation please adjust the following parameters:\n"
                      "Selected ids: Comma-separated Organoid IDs for annotation (if empty will use all organoids from selected layer; can select range of ids with \'-\'e.g. 1-10)\n"
                      "Padding: Padding around each bounding box in organoid visualization\n"
                      "Bounding box width: Width of detection bounding box in organoid visualization"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Selected ids
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Selected organoid ids:"))
        self.selected_ids_edit = QLineEdit()
        hbox.addWidget(self.selected_ids_edit)
        self.selected_ids_edit.setText(self.id_str)
        layout.addLayout(hbox)
        
        # Padding field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Padding (pixels):"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(self.padding)
        hbox.addWidget(self.padding_spin)
        layout.addLayout(hbox)

        #Border width for bbox visualization
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Bounding box width (pixels):"))
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(self.border_width)
        hbox.addWidget(self.border_width_spin)
        layout.addLayout(hbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(super().reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_annotation)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _setup_annotation_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Image and properties layout
        content_layout = QHBoxLayout()
        
        # Image display
        self.image_label = RulerImageLabel(dialogue_class=self)
        content_layout.addWidget(self.image_label, 2)
        
        # Properties table
        self.props_table = QTableWidget()
        self.props_table.setColumnCount(2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.props_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content_layout.addWidget(self.props_table, 1)
        
        layout.addLayout(content_layout, 3)

        self.progress_label = QLabel("Annotation progress: ")
        layout.addWidget(self.progress_label)

        self.coord_label = QLabel("Current coordinates: None")
        layout.addWidget(self.coord_label)

        
        # Annotation field
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotation (List of lines vertices):"))
        self.annotation_edit = QLineEdit()
        self.annotation_edit.setReadOnly(True)
        self.annotation_edit.returnPressed.connect(self.show_next)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.clicked.connect(self.image_label.clear_image)
        self.clear_last_btn = QPushButton("Clear last segment")
        self.clear_last_btn.clicked.connect(self.image_label.clear_last_segment)
        annot_layout.addWidget(self.annotation_edit)
        annot_layout.addWidget(self.clear_btn)
        annot_layout.addWidget(self.clear_last_btn)

        # Total length
        tl_layout = QHBoxLayout()
        tl_layout.addWidget(QLabel("Total length:"))
        self.annotation_total_length = QLineEdit()
        self.annotation_total_length.setReadOnly(True)
        self.annotation_total_length.setValidator(QDoubleValidator())
        tl_layout.addWidget(self.annotation_total_length)

        # Average length
        avg_len_layout = QHBoxLayout()
        avg_len_layout.addWidget(QLabel("Average length:"))
        self.annotation_average_length = QLineEdit()
        self.annotation_average_length.setReadOnly(True)
        self.annotation_average_length.setValidator(QDoubleValidator())
        avg_len_layout.addWidget(self.annotation_average_length)

        # Total count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Lines count:"))
        self.annotation_count = QLineEdit()
        self.annotation_count.setReadOnly(True)
        self.annotation_count.setValidator(QDoubleValidator())
        count_layout.addWidget(self.annotation_count)
        
        layout.addLayout(annot_layout)
        layout.addLayout(tl_layout)
        layout.addLayout(avg_len_layout)
        layout.addLayout(count_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.cancel_btn2 = QPushButton("Cancel")
        self.cancel_btn2.clicked.connect(self.reject)
        nav_layout.addWidget(self.cancel_btn2)

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.show_prev)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.accept)
        nav_layout.addWidget(self.finish_btn)
        
        layout.addLayout(nav_layout)
        
        return widget
    
    def start_annotation(self):
        """Initialize annotation process"""
        # Get parameters from setup screen
        self.padding = self.padding_spin.value()
        self.border_width = self.border_width_spin.value()

        def get_error_dialog(text):
            return QMessageBox.warning(self, "Invalid IDs", f"Invalid selected IDs string ({text}).")

        if len(self.selected_ids_edit.text().strip()) > 0:
            self.annotated_ids = set()
            for token in self.selected_ids_edit.text().strip().split(','):
                token = token.strip()
                if token == "":
                    get_error_dialog(f"Empty token encountered")
                    return
                if '-' in token:
                    range_data = token.split('-')
                    if len(range_data) != 2:
                        get_error_dialog(f"Invalid range {token}")
                        return
                    try:
                        start = int(range_data[0])
                        end = int(range_data[1])
                        if start < 0 or end < 0 or start > end:
                            get_error_dialog(f"Invalid range {token}")
                            return
                        for curr_id in range(start, end+1):
                            if not curr_id in self.layer_properties['bbox_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['bbox_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['bbox_id'])
        
        self.annotations = {key: val for key, val in self.annotations.items() if int(key) in self.annotated_ids}
        self.annotated_ids = list(self.annotated_ids)
        
        # Switch to annotation screen
        self.stacked_widget.setCurrentIndex(1)
        
        # Initialize annotation state
        self.current_idx = 0
        self.update_display()
    
    def update_display(self):
        """Update displayed bounding box"""
        box_id = self.annotated_ids[self.current_idx]
        layer_data_id = np.where(self.layer_properties['bbox_id'] == box_id)[0][0]
        x1, y1, x2, y2 = self.layer_data[layer_data_id]
        
        h, w = self.image.height(), self.image.width()
        x1_disp = max(0, int(x1) - self.padding)
        y1_disp = max(0, int(y1) - self.padding)
        x2_disp = min(h, int(x2) + self.padding)
        y2_disp = min(w, int(y2) + self.padding)
        
        # Display image snippet
        snippet = self.image.copy(y1_disp, x1_disp, y2_disp - y1_disp, x2_disp - x1_disp)
        cur_snippet_coords = [y1_disp, x1_disp]
        self.image_label.set_snippet_coords(cur_snippet_coords)

        painter = QPainter(snippet)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(self.border_width)
        painter.setPen(pen)
        painter.drawRect(
            self.padding + self.border_width // 2,
            self.padding + self.border_width // 2,
            int(y2 - y1 - self.border_width),
            int(x2 - x1 - self.border_width),
        )
        painter.end()

        pixmap = QPixmap.fromImage(snippet)
        self.image_label.setPixmap(pixmap)
        
        self.props_table.setRowCount(len(self.layer_properties))
        for i, (prop, values) in enumerate(self.layer_properties.items()):
            value = values[layer_data_id]
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.props_table.item(i, 0).setToolTip(str(prop))
            self.props_table.item(i, 1).setToolTip(str(value))    
            
        # Set current annotation
        if str(box_id) in self.annotations:
            current_annot = self.annotations[str(box_id)]
        else:
            current_annot = []
        self.annotation_edit.setText(str(current_annot))
        self.image_label.set_lines(current_annot)
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)
        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def save_annotation_config(self):
        """Save current annotation to dictionary"""
        if self.current_idx >= 0:
            box_id = self.annotated_ids[self.current_idx]
            self.annotations[str(box_id)] = self.image_label.get_lines()
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'property_name': self.property_name,
            'type': "Ruler",
            # 'data': self.annotations,
            # 'ids': self.selected_ids_edit.text().strip(),
            'padding': self.padding,
            'border_width': self.border_width
        }
        session.set_session_var('annotation_features', annotation_features)

    def save_annotation(self):
        """Save current annotation to dictionary"""
        if self.current_idx >= 0:
            box_id = self.annotated_ids[self.current_idx]
            self.annotations[str(box_id)] = self.image_label.get_lines()
        
        # Update shape layer properties with annotated values
        data = self.annotations
        ids = self.selected_ids_edit.text().strip(),
    
    def show_prev(self):
        """Show previous bounding box"""
        self.save_annotation()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
        else:
            self.current_idx = -1
            self.cur_snippet_coords = None
            self.stacked_widget.setCurrentIndex(0)
    
    def show_next(self):
        """Show next bounding box"""
        self.save_annotation()
        if self.current_idx < len(self.annotated_ids) - 1:
            self.current_idx += 1
            self.update_display()
    
    def get_annotations(self):
        """Return the updated annotations dictionary"""
        final_annot = {}
        for box_id, val in self.annotations.items():
            total_length, average_length, count = get_poly_lines_data(val)
            final_annot[box_id] = (str(val), total_length, average_length, count)
        return final_annot
    
    def accept(self):
        self.save_annotation()
        super().accept()

    def reject(self):
        self.save_annotation()
        super().reject()

class ClassAnnotationDialogue(AnnotationDialogue):
    def __init__(self, image, layer_data, layer_properties, annotation_data, parent=None):
        self.classes_list = set(annotation_data.get('classes_list', []))
        super().__init__(image, layer_data, layer_properties, annotation_data, parent)

    def _setup_start_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc = QLabel("INFO: This tool is for annotating classes on existing label layers. You can adjust list of available classes on this window."
                      "During annotation you could select 1 or more classes for every detection. Before starting annotation please adjust the following parameters:\n"
                      "Selected ids: Comma-separated Organoid IDs for annotation (if empty will use all organoids from selected layer; can select range of ids with \'-\' e.g. 1-10)\n"
                      "Padding: Padding around each bounding box in organoid visualization\n"
                      "Bounding box width: Width of detection bounding box in organoid visualization"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        title = QLabel("List of current classes:")
        title.setAlignment(Qt.AlignHCenter)
        layout.addWidget(title)

        self.class_table = QTableWidget()
        self.class_table.setColumnCount(2)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.horizontalHeader().setVisible(False)
        self.class_table.verticalHeader().setDefaultSectionSize(50)
        self.class_table.verticalHeader().setMinimumSectionSize(50)
        self.class_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.class_table.setShowGrid(False)
        self.class_table.setRowCount(len(self.classes_list))
        self.fill_class_table()
        layout.addWidget(self.class_table)

        add_layout = QHBoxLayout()
        add_layout.addWidget(QLabel("New class: "))
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("Enter new class name")
        add_layout.addWidget(self.new_class_edit)
        add_btn = QPushButton("Add Class")
        add_btn.clicked.connect(self.add_class)
        add_layout.addWidget(add_btn)
        layout.addLayout(add_layout)


        # Selected ids
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Selected organoid ids:"))
        self.selected_ids_edit = QLineEdit()
        hbox.addWidget(self.selected_ids_edit)
        self.selected_ids_edit.setText(self.id_str)
        layout.addLayout(hbox)
        
        # Padding field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Padding (pixels):"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(self.padding)
        hbox.addWidget(self.padding_spin)
        layout.addLayout(hbox)

        #Border width for bbox visualization
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Bounding box width (pixels):"))
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(self.border_width)
        hbox.addWidget(self.border_width_spin)
        layout.addLayout(hbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(super().reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_annotation)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def fill_class_table(self):
        self.class_table.setRowCount(len(self.classes_list))
        for row, class_name in enumerate(self.classes_list):
            class_item = QTableWidgetItem(class_name)
            self.class_table.setItem(row, 0, class_item)
            delete_btn = QPushButton("Delete")
            delete_btn.setMaximumHeight(30)
            delete_btn.setMaximumWidth(60)
            delete_btn.clicked.connect(lambda _, r=row: self.delete_class(r))
            self.class_table.setCellWidget(row, 1, delete_btn)
    
    def add_class(self):
        class_name = self.new_class_edit.text().strip()
        if len(class_name) == 0:
            QMessageBox.warning(self, "Invalid class", "Empty class name")
        if class_name in self.classes_list:
            QMessageBox.warning(self, "Invalid class", f"Class of name \"{class_name}\" already exists")
        self.classes_list.add(class_name)
        self.new_class_edit.clear()
        self.fill_class_table()

    def delete_class(self, row):
        class_name = self.class_table.item(row, 0).text()
        if class_name in self.classes_list:
            self.classes_list.remove(class_name)
        self.fill_class_table()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'scroll_area'):
            max_height = int(self.height() * 0.3)
            self.scroll_area.setFixedHeight(max_height)
    
    def _setup_annotation_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Image and properties layout
        content_layout = QHBoxLayout()
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.image_label.setMinimumSize(300, 300)
        content_layout.addWidget(self.image_label, 2)
        
        # Properties table
        self.props_table = QTableWidget()
        self.props_table.setColumnCount(2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.props_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content_layout.addWidget(self.props_table, 1)
        
        layout.addLayout(content_layout, 3)

        self.progress_label = QLabel("Annotation progress: ")
        layout.addWidget(self.progress_label)  

        title = QLabel("Annotated classes:")
        title.setAlignment(Qt.AlignHCenter)
        layout.addWidget(title)

        # Add checkboxes here
        self.scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.selector_layout = QVBoxLayout(scroll_widget)
        self.scroll_area.setWidget(scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.cancel_btn2 = QPushButton("Cancel")
        self.cancel_btn2.clicked.connect(self.reject)
        nav_layout.addWidget(self.cancel_btn2)

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.show_prev)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.accept)
        nav_layout.addWidget(self.finish_btn)
        
        layout.addLayout(nav_layout)
        
        return widget
    
    def start_annotation(self):
        self.padding = self.padding_spin.value()
        self.border_width = self.border_width_spin.value()

        def get_error_dialog(text):
            return QMessageBox.warning(self, "Invalid IDs", f"Invalid selected IDs string ({text}).")

        if len(self.selected_ids_edit.text().strip()) > 0:
            self.annotated_ids = set()
            for token in self.selected_ids_edit.text().strip().split(','):
                token = token.strip()
                if token == "":
                    get_error_dialog(f"Empty token encountered")
                    return
                if '-' in token:
                    range_data = token.split('-')
                    if len(range_data) != 2:
                        get_error_dialog(f"Invalid range {token}")
                        return
                    try:
                        start = int(range_data[0])
                        end = int(range_data[1])
                        if start < 0 or end < 0 or start > end:
                            get_error_dialog(f"Invalid range {token}")
                            return
                        for curr_id in range(start, end+1):
                            if not curr_id in self.layer_properties['bbox_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['bbox_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['bbox_id'])
        
        self.annotations = {key: list(val) for key, val in self.annotations.items() if int(key) in self.annotated_ids}

        self.annotated_ids = list(self.annotated_ids)
        
        self.stacked_widget.setCurrentIndex(1)
        
        self.current_idx = 0
        self.update_display()        
    
    def update_display(self):
        """Update displayed bounding box"""
        self.clear_class_selectors()
        box_id = self.annotated_ids[self.current_idx]
        layer_data_id = np.where(self.layer_properties['bbox_id'] == box_id)[0][0]
        x1, y1, x2, y2 = self.layer_data[layer_data_id]
        
        h, w = self.image.height(), self.image.width()
        x1_disp = max(0, int(x1) - self.padding)
        y1_disp = max(0, int(y1) - self.padding)
        x2_disp = min(h, int(x2) + self.padding)
        y2_disp = min(w, int(y2) + self.padding)
        
        # Display image snippet
        snippet = self.image.copy(y1_disp, x1_disp, y2_disp - y1_disp, x2_disp - x1_disp)

        painter = QPainter(snippet)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(self.border_width)
        painter.setPen(pen)
        painter.drawRect(
            self.padding + self.border_width // 2,
            self.padding + self.border_width // 2,
            int(y2 - y1 - self.border_width),
            int(x2 - x1 - self.border_width),
        )
        painter.end()

        pixmap = QPixmap.fromImage(snippet)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        self.props_table.setRowCount(len(self.layer_properties))
        for i, (prop, values) in enumerate(self.layer_properties.items()):
            value = values[layer_data_id]
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.props_table.item(i, 0).setToolTip(str(prop))
            self.props_table.item(i, 1).setToolTip(str(value))

        for class_name in self.classes_list:
            checkbox = QCheckBox(class_name)
            self.selector_layout.addWidget(checkbox)

        if str(box_id) in self.annotations:
            classes = self.annotations[str(box_id)]
            for i in range(self.selector_layout.count()):
                checkbox = self.selector_layout.itemAt(i).widget()
                if checkbox and isinstance(checkbox, QCheckBox):
                    checkbox.setChecked(checkbox.text() in classes)
            
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)
        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def clear_class_selectors(self):
        """Clear all checkboxes from the selector layout"""
        while self.selector_layout.count():
            child = self.selector_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def get_annotated_classes(self):
        """Return list of selected class labels from checkboxes"""
        selected_classes = []
        for i in range(self.selector_layout.count()):
            checkbox = self.selector_layout.itemAt(i).widget()
            if checkbox and isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                selected_classes.append(checkbox.text())
        return selected_classes
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
        if self.current_idx >= 0:
            box_id = self.annotated_ids[self.current_idx]
            self.annotations[str(box_id)] = self.get_annotated_classes()
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'property_name': self.property_name,
            'type': "Classes",
            'data': self.annotations,
            'ids': self.selected_ids_edit.text().strip(),
            'classes_list': list(self.classes_list),
            'padding': self.padding,
            'border_width': self.border_width
        }
        session.set_session_var('annotation_features', annotation_features)

    
    def show_prev(self):
        """Show previous bounding box"""
        self.save_annotation()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
        else:
            self.current_idx = -1
            self.stacked_widget.setCurrentIndex(0)
    
    def show_next(self):
        """Show next bounding box"""
        self.save_annotation()
        if self.current_idx < len(self.annotated_ids) - 1:
            self.current_idx += 1
            self.update_display()
    
    def get_annotations(self):
        """Return the updated annotations dictionary"""
        return {key: ','.join(val) for key, val in self.annotations.items()}
    
    def accept(self):
        self.save_annotation()
        super().accept()

    def reject(self):
        self.save_annotation()
        super().reject()

class BboxImageLabel(QLabel):
    """Custom QLabel for bounding box annotation tool"""
    def __init__(self, dialogue_class, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self._pixmap = None
        self._original_pixmap = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.line_color = QColor(255, 0, 0)
        self.line_width = 2
        self.dialogue_class = dialogue_class
        self.bboxes = []
        self.cur_snippet_coords = None

    def set_snippet_coords(self, coords):
        """Set global coordinates of the current image snippet's top-left corner"""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ValueError("Snippet coordinates must be a list or tuple of two elements (y, x).")
        self.cur_snippet_coords = coords

    def setPixmap(self, pixmap):
        """Store the original pixmap and update the display"""
        self._original_pixmap = pixmap
        self._pixmap = pixmap.copy() if pixmap else None
        self.clear_image()
        super().setPixmap(self.scaled_pixmap())
        
    def scaled_pixmap(self):
        """Return scaled pixmap based on current label size"""
        if self._pixmap is None or self._pixmap.isNull():
            return QPixmap()
        pixmap_size = self._pixmap.size()
        pixmap_size.scale(self.size(), Qt.IgnoreAspectRatio)
        return self._pixmap.scaled(pixmap_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        
    def resizeEvent(self, event):
        """Handle resize events to update the pixmap"""
        if self._pixmap and not self._pixmap.isNull():
            super().setPixmap(self.scaled_pixmap())
        super().resizeEvent(event)
        
    def mousePressEvent(self, event):
        """Start drawing a new line"""
        if event.button() == Qt.LeftButton:
            self.start_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            self.end_point = self.start_point
            self.drawing = True
            self.bboxes.append(self._adjust_points())
            self.update_bboxes()
            
    def mouseMoveEvent(self, event):
        """Update line while dragging"""
        if self.drawing:
            self.end_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            self.bboxes[-1] = self._adjust_points()
            self.update_bboxes()
            
    def mouseReleaseEvent(self, event):
        """Finish drawing line"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            self.bboxes[-1] = self._adjust_points()
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.update_bboxes()

    def _adjust_points(self):
        if self.start_point is None or self.end_point is None:
            raise ValueError("Start and end points must be set before adjusting points.")
        top_left = QPoint(min(self.start_point.x(), self.end_point.x()),
                              min(self.start_point.y(), self.end_point.y()))
        bottom_right = QPoint(max(self.start_point.x(), self.end_point.x()),
                              max(self.start_point.y(), self.end_point.y()))
        return [top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y()]

    def get_bboxes(self):
        if self.cur_snippet_coords is None:
            raise ValueError("Snippet coordinates must be set before getting bounding boxes.")
        global_bboxes = copy.deepcopy(self.bboxes)
        for bbox in global_bboxes:
            bbox[0] += self.cur_snippet_coords[0]
            bbox[1] += self.cur_snippet_coords[1]
            bbox[2] += self.cur_snippet_coords[0]
            bbox[3] += self.cur_snippet_coords[1]
            tmp = bbox[0]
            bbox[0] = bbox[1]
            bbox[1] = tmp
            tmp = bbox[2]
            bbox[2] = bbox[3]
            bbox[3] = tmp
        return global_bboxes
    
    def set_bboxes(self, bboxes):
        if self.cur_snippet_coords is None:
            raise ValueError("Snippet coordinates must be set before setting bounding boxes.")
        self.bboxes = []
        for bbox in bboxes:
            tmp = bbox[0]
            bbox[0] = bbox[1]
            bbox[1] = tmp
            tmp = bbox[2]
            bbox[2] = bbox[3]
            bbox[3] = tmp
            bbox[0] -= self.cur_snippet_coords[0]
            bbox[1] -= self.cur_snippet_coords[1]
            bbox[2] -= self.cur_snippet_coords[0]
            bbox[3] -= self.cur_snippet_coords[1]
            self.bboxes.append(bbox)
        self.update_bboxes()
            
    def clear_image(self):
        """Clear the current ruler line"""
        self.bboxes = []
        self.update_bboxes()

    def clear_last_bbox(self):
        """Clear last drawn bbox"""
        if len(self.bboxes):
            self.bboxes = self.bboxes[:-1]
            self.update_bboxes()
            
    def update_bboxes(self):
        """Redraw bboxes on the image"""
        if self._original_pixmap:
            self._pixmap = self._original_pixmap.copy()
            painter = QPainter(self._pixmap)
            pen = QPen(self.line_color)
            pen.setWidth(self.line_width)
            painter.setPen(pen)

            for bbox in self.bboxes:
                painter.drawRect(bbox[0], bbox[1],
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1])           
            painter.end()
            super().setPixmap(self.scaled_pixmap())
            self.dialogue_class.annotation_edit.setText(str(self.get_bboxes()))

    def original_to_scaled(self, point):
        """Convert a point from original image coordinates to label coordinates"""
        label_size = self.size()
        pixmap_size = self._original_pixmap.size()

        scale_factor_x = label_size.width() / pixmap_size.width()
        scale_factor_y = label_size.height() / pixmap_size.height()
        
        scaled_x = point.x() * scale_factor_x
        scaled_y = point.y() * scale_factor_y
        
        return QPoint(int(scaled_x), int(scaled_y))
            
    def scaled_to_original(self, point):
        """Convert a point from label coordinates to original image coordinates"""
        label_size = self.size()
        pixmap_size = self._original_pixmap.size()
        
        scale_factor_x = label_size.width() / pixmap_size.width()
        scale_factor_y = label_size.height() / pixmap_size.height()
        
        orig_x = point.x() / scale_factor_x
        orig_y = point.y() / scale_factor_y
        
        return QPoint(int(orig_x), int(orig_y))
        
class BboxAnnotationWidget(AnnotationDialogue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_start_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc = QLabel("INFO: This tool is for annotating properties with ruler tool. You could draw a single line on every image. The feature will be populated with line length."
                      "Before starting annotation please adjust the following parameters:\n"
                      "Selected ids: Comma-separated Organoid IDs for annotation (if empty will use all organoids from selected layer; can select range of ids with \'-\'e.g. 1-10)\n"
                      "Padding: Padding around each bounding box in organoid visualization\n"
                      "Bounding box width: Width of detection bounding box in organoid visualization"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Selected ids
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Selected organoid ids:"))
        self.selected_ids_edit = QLineEdit()
        hbox.addWidget(self.selected_ids_edit)
        self.selected_ids_edit.setText(self.id_str)
        layout.addLayout(hbox)
        
        # Padding field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Padding (pixels):"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(self.padding)
        hbox.addWidget(self.padding_spin)
        layout.addLayout(hbox)

        #Border width for bbox visualization
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Bounding box width (pixels):"))
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(self.border_width)
        hbox.addWidget(self.border_width_spin)
        layout.addLayout(hbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(super().reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_annotation)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _setup_annotation_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Image and properties layout
        content_layout = QHBoxLayout()
        
        # Image display
        self.image_label = BboxImageLabel(dialogue_class=self)
        content_layout.addWidget(self.image_label, 2)
        
        # Properties table
        self.props_table = QTableWidget()
        self.props_table.setColumnCount(2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.props_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content_layout.addWidget(self.props_table, 1)
        
        layout.addLayout(content_layout, 3)

        self.progress_label = QLabel("Annotation progress: ")
        layout.addWidget(self.progress_label)
        
        # Annotation field
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotation (List of bboxes corners):"))
        self.annotation_edit = QLineEdit()
        self.annotation_edit.setReadOnly(True)
        self.annotation_edit.returnPressed.connect(self.show_next)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.clicked.connect(self.image_label.clear_image)
        self.clear_last_btn = QPushButton("Clear last")
        self.clear_last_btn.clicked.connect(self.image_label.clear_last_bbox)
        annot_layout.addWidget(self.annotation_edit)
        annot_layout.addWidget(self.clear_btn)
        annot_layout.addWidget(self.clear_last_btn)
        
        layout.addLayout(annot_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.cancel_btn2 = QPushButton("Cancel")
        self.cancel_btn2.clicked.connect(self.reject)
        nav_layout.addWidget(self.cancel_btn2)

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.show_prev)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.accept)
        nav_layout.addWidget(self.finish_btn)
        
        layout.addLayout(nav_layout)
        
        return widget
    
    def start_annotation(self):
        """Initialize annotation process"""
        # Get parameters from setup screen
        self.padding = self.padding_spin.value()
        self.border_width = self.border_width_spin.value()

        def get_error_dialog(text):
            return QMessageBox.warning(self, "Invalid IDs", f"Invalid selected IDs string ({text}).")

        if len(self.selected_ids_edit.text().strip()) > 0:
            self.annotated_ids = set()
            for token in self.selected_ids_edit.text().strip().split(','):
                token = token.strip()
                if token == "":
                    get_error_dialog(f"Empty token encountered")
                    return
                if '-' in token:
                    range_data = token.split('-')
                    if len(range_data) != 2:
                        get_error_dialog(f"Invalid range {token}")
                        return
                    try:
                        start = int(range_data[0])
                        end = int(range_data[1])
                        if start < 0 or end < 0 or start > end:
                            get_error_dialog(f"Invalid range {token}")
                            return
                        for curr_id in range(start, end+1):
                            if not curr_id in self.layer_properties['bbox_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['bbox_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['bbox_id'])
        
        self.annotations = {key: val for key, val in self.annotations.items() if int(key) in self.annotated_ids}
        self.annotated_ids = list(self.annotated_ids)
        
        # Switch to annotation screen
        self.stacked_widget.setCurrentIndex(1)
        
        # Initialize annotation state
        self.current_idx = 0
        self.update_display()
    
    def update_display(self):
        """Update displayed bounding box"""
        box_id = self.annotated_ids[self.current_idx]
        layer_data_id = np.where(self.layer_properties['bbox_id'] == box_id)[0][0]
        x1, y1, x2, y2 = self.layer_data[layer_data_id]
        
        h, w = self.image.height(), self.image.width()
        x1_disp = max(0, int(x1) - self.padding)
        y1_disp = max(0, int(y1) - self.padding)
        x2_disp = min(h, int(x2) + self.padding)
        y2_disp = min(w, int(y2) + self.padding)
        
        # Display image snippet
        snippet = self.image.copy(y1_disp, x1_disp, y2_disp - y1_disp, x2_disp - x1_disp)
        cur_snippet_coords = [y1_disp, x1_disp]
        self.image_label.set_snippet_coords(cur_snippet_coords)

        painter = QPainter(snippet)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(self.border_width)
        painter.setPen(pen)
        painter.drawRect(
            self.padding + self.border_width // 2,
            self.padding + self.border_width // 2,
            int(y2 - y1 - self.border_width),
            int(x2 - x1 - self.border_width),
        )
        painter.end()

        pixmap = QPixmap.fromImage(snippet)
        self.image_label.setPixmap(pixmap)
        
        self.props_table.setRowCount(len(self.layer_properties))
        for i, (prop, values) in enumerate(self.layer_properties.items()):
            value = values[layer_data_id]
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.props_table.item(i, 0).setToolTip(str(prop))
            self.props_table.item(i, 1).setToolTip(str(value))    
            
        # Set current annotation
        if str(box_id) in self.annotations:
            current_annot = self.annotations[str(box_id)]
        else:
            current_annot = []
        self.annotation_edit.setText(str(current_annot))
        self.image_label.set_bboxes(current_annot)
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)
        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
        if self.current_idx >= 0:
            box_id = self.annotated_ids[self.current_idx]
            self.annotations[str(box_id)] = self.image_label.get_bboxes()
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'property_name': self.property_name,
            'type': "Objects / Boxes",
            'data': self.annotations,
            'ids': self.selected_ids_edit.text().strip(),
            'padding': self.padding,
            'border_width': self.border_width
        }
        session.set_session_var('annotation_features', annotation_features)

    
    def show_prev(self):
        """Show previous bounding box"""
        self.save_annotation()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
        else:
            self.current_idx = -1
            self.cur_snippet_coords = None
            self.stacked_widget.setCurrentIndex(0)
    
    def show_next(self):
        """Show next bounding box"""
        self.save_annotation()
        if self.current_idx < len(self.annotated_ids) - 1:
            self.current_idx += 1
            self.update_display()
    
    def get_annotations(self):
        """Return the updated annotations dictionary"""
        return {box_id: str(val) for box_id, val in self.annotations.items()}
    
    def accept(self):
        self.save_annotation()
        super().accept()

    def reject(self):
        self.save_annotation()
        super().reject()