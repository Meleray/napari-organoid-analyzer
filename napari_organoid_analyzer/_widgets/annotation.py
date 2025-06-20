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
    QScrollArea
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

def get_annotation_dialogue(image, layer_data, layer_properties, annotation_data, parent):
    type = annotation_data['type']
    if type == "Text":
        return TextAnnotationDialogue(image, layer_data, layer_properties, annotation_data, parent)
    elif type == "Ruler":
        return RulerAnnotationDialogue(image, layer_data, layer_properties, annotation_data, parent)
    elif type == 'Objects / Boxes':
        pass
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
                            if not curr_id in self.layer_properties['box_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['box_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['box_id'])
        
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
        layer_data_id = np.where(self.layer_properties['box_id'] == box_id)[0][0]
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
        self.default_value_edit.setText(self.default_value)
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
                            if not curr_id in self.layer_properties['box_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['box_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['box_id'])
        
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
        layer_data_id = np.where(self.layer_properties['box_id'] == box_id)[0][0]
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

class RulerImageLabel(QLabel):
    """Custom QLabel for ruler annotation tool"""
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
        
    def setPixmap(self, pixmap):
        """Store the original pixmap and update the display"""
        self._original_pixmap = pixmap
        self._pixmap = pixmap.copy() if pixmap else None
        self.clear_line()
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
            self.update_line()
            
    def mouseMoveEvent(self, event):
        """Update line while dragging"""
        if self.drawing:
            self.end_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            self.update_line()
            
    def mouseReleaseEvent(self, event):
        """Finish drawing line"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = self.scaled_to_original(self.mapFromGlobal(event.globalPos()))
            self.drawing = False
            self.update_line()

    def get_line(self):
        if self.start_point is None or self.end_point is None:
            return None
        return [self.start_point.x(), self.start_point.y(), self.end_point.x(), self.end_point.y()]
    
    def set_line(self, line):
        self.start_point = QPoint(line[0], line[1])
        self.end_point = QPoint(line[2], line[3])
        self.update_line()
            
    def clear_line(self):
        """Clear the current ruler line"""
        self.start_point = None
        self.end_point = None
        self.drawing = False
        if self._original_pixmap:
            self._pixmap = self._original_pixmap.copy()
            super().setPixmap(self.scaled_pixmap())
            
    def update_line(self):
        """Update the display with the current line"""
        if self.start_point is None or self.end_point is None:
            return
            
        if self._original_pixmap:
            self._pixmap = self._original_pixmap.copy()
            painter = QPainter(self._pixmap)
            distance = get_length(self.start_point.x(), self.start_point.y(), self.end_point.x(), self.end_point.y())
            self.dialogue_class.annotation_edit.setText(f"{distance:.4f}")
            pen = QPen(self.line_color)
            pen.setWidth(self.line_width)
            painter.setPen(pen)
            painter.drawLine(self.start_point, self.end_point)
            
            painter.end()
            super().setPixmap(self.scaled_pixmap())

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
        self.cur_snippet_coords = None

    
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

        
        # Annotation field
        annot_layout = QHBoxLayout()
        annot_layout.addWidget(QLabel("Annotation (length in pixels):"))
        self.annotation_edit = QLineEdit()
        validator = QDoubleValidator()
        validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.annotation_edit.setValidator(validator)
        self.annotation_edit.setReadOnly(True)
        self.annotation_edit.returnPressed.connect(self.show_next)
        self.clear_btn = QPushButton("Clear Line")
        self.clear_btn.clicked.connect(self.image_label.clear_line)
        annot_layout.addWidget(self.annotation_edit)
        annot_layout.addWidget(self.clear_btn)
        
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
                            if not curr_id in self.layer_properties['box_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['box_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['box_id'])
        
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
        layer_data_id = np.where(self.layer_properties['box_id'] == box_id)[0][0]
        x1, y1, x2, y2 = self.layer_data[layer_data_id]
        
        h, w = self.image.height(), self.image.width()
        x1_disp = max(0, int(x1) - self.padding)
        y1_disp = max(0, int(y1) - self.padding)
        x2_disp = min(h, int(x2) + self.padding)
        y2_disp = min(w, int(y2) + self.padding)
        
        # Display image snippet
        snippet = self.image.copy(y1_disp, x1_disp, y2_disp - y1_disp, x2_disp - x1_disp)
        self.cur_snippet_coords = [y1_disp, x1_disp]

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
            old_line_coords = self.annotations[str(box_id)]
            old_line_coords[0] -= self.cur_snippet_coords[0]
            old_line_coords[1] -= self.cur_snippet_coords[1]
            old_line_coords[2] -= self.cur_snippet_coords[0]
            old_line_coords[3] -= self.cur_snippet_coords[1]
            current_annot = get_length(*old_line_coords)
            self.image_label.set_line(old_line_coords)
        else:
            current_annot = 0.0
        self.annotation_edit.setText(f"{current_annot:.4f}")
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)
        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
        box_id = self.annotated_ids[self.current_idx]
        cur_line = self.image_label.get_line()
        if cur_line is not None:
            # TODO: Maybe other way around
            cur_line[0] += self.cur_snippet_coords[0]
            cur_line[1] += self.cur_snippet_coords[1]
            cur_line[2] += self.cur_snippet_coords[0]
            cur_line[3] += self.cur_snippet_coords[1]
            self.annotations[str(box_id)] = cur_line
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'property_name': self.property_name,
            'type': "Ruler",
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
        return {box_id: get_length(*val) for box_id, val in self.annotations.items()}
    
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

        add_selector_btn = QPushButton("+ Add Another Class")
        add_selector_btn.clicked.connect(self.add_class_selector)
        layout.addWidget(add_selector_btn)

        self.selector_layout = QVBoxLayout()
        layout.addLayout(self.selector_layout)
        
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
    
    def add_class_selector(self, cls=""):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        
        combo = QComboBox()
        combo.addItems(list(self.classes_list))
        combo.setCurrentText("")
        combo.currentIndexChanged.connect(self.save_annotation)
        row_layout.addWidget(combo, 1)
        
        delete_btn = QPushButton("Remove")
        delete_btn.setFixedWidth(80)
        delete_btn.clicked.connect(lambda: self.remove_class_selector(row_widget))
        row_layout.addWidget(delete_btn)
        
        self.selector_layout.addWidget(row_widget)

        if cls in self.classes_list:
            combo.setCurrentText(cls)
            combo.setCurrentIndex(combo.findText(cls))

    def remove_class_selector(self, selector_row):
        for i in range(self.selector_layout.count()):
            widget = self.selector_layout.itemAt(i).widget()
            if widget == selector_row:
                widget.deleteLater()
                break

    def get_annotated_classes(self):
        classes = set()
        for i in range(self.selector_layout.count()):
            widget = self.selector_layout.itemAt(i).widget()
            if widget:
                combo = widget.findChild(QComboBox)
                if combo and combo.currentText():
                    classes.add(combo.currentText())
        if len(classes):
            return list(classes)
        else:
            return []
    
    def clear_class_selectors(self):
        for i in range(self.selector_layout.count()):
            widget = self.selector_layout.itemAt(i).widget()
            widget.deleteLater()   
    
    def start_annotation(self):
        """Initialize annotation process"""
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
                            if not curr_id in self.layer_properties['box_id']:
                                get_error_dialog(f"ID {curr_id} not found in labels")
                                return
                            self.annotated_ids.add(curr_id)

                    except ValueError:
                        get_error_dialog(f"Invalid range {token}")
                        return
                else:
                    try:
                        curr_id = int(token)
                        if not curr_id in self.layer_properties['box_id']:
                            get_error_dialog(f"ID {curr_id} not found in labels")
                            return
                    except ValueError:
                        get_error_dialog(f"Invalid token \"{token}\"")
                        return 
                    self.annotated_ids.add(curr_id)  
        else:
            self.annotated_ids = set(self.layer_properties['box_id'])
        
        self.annotations = {key: list(val) for key, val in self.annotations.items() if int(key) in self.annotated_ids}

        self.annotated_ids = list(self.annotated_ids)
        
        self.stacked_widget.setCurrentIndex(1)
        
        self.current_idx = 0
        self.update_display()        
    
    def update_display(self):
        """Update displayed bounding box"""
        self.clear_class_selectors()
        box_id = self.annotated_ids[self.current_idx]
        layer_data_id = np.where(self.layer_properties['box_id'] == box_id)[0][0]
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

        if str(box_id) in self.annotations:
            classes = self.annotations[str(box_id)]
            for cls in classes:
                if cls in self.classes_list:
                    self.add_class_selector(cls)
            
        self.next_btn.setEnabled(self.current_idx < len(self.annotated_ids) - 1)
        self.progress_label.setText(f"Annotation progress: {self.current_idx + 1}/{len(self.annotated_ids)}")
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
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