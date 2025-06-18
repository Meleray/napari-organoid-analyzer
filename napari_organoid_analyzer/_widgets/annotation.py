from napari_organoid_analyzer import session
from napari_organoid_analyzer import settings
from qtpy.QtCore import Qt
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
    QStackedWidget
)
from qtpy.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen
)

def get_annotation_dialogue(image, layer_data, layer_properties, annotation_data):
    type = annotation_data['type']
    if type == "Text":
        return TextAnnotationDialogue(image, layer_data, layer_properties, annotation_data)
    elif type == "Ruler":
        pass
    elif type == 'Objects / Boxes':
        pass
    elif type == 'Classes':
        pass
    elif type == 'Number':
        pass
    else:
        raise RuntimeError(f"Unknown annotation type {type}!")
    
class AnnotationDialogue(QDialog):
    def __init__(self, image, layer_data, layer_properties, annotation_data, parent=None):
        super().__init__(parent)
        self.layer_data = layer_data
        self.layer_properties = layer_properties
        self.annotations = annotation_data['data']
        self.annotation_name = annotation_data['annotation_name']
        self.property_name = annotation_data['property_name']

        if image.ndim == 2:
            h, w = image.shape
            bytes_per_line = w
            self.image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif image.ndim == 3:
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
        self.resize(800, 600)

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
        
        desc = QLabel("INFO: This tool is for annotating text properties on existing label layers.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Default value field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Default Annotation Value:"))
        self.default_value_edit = QLineEdit()
        hbox.addWidget(self.default_value_edit)
        layout.addLayout(hbox)
        
        # Padding field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Padding (pixels):"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 100)
        self.padding_spin.setValue(10)
        hbox.addWidget(self.padding_spin)
        layout.addLayout(hbox)

        #Border width for bbox visualization
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Bounding box width (pixels):"))
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(2)
        hbox.addWidget(self.border_width_spin)
        layout.addLayout(hbox)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
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
        for box_id in self.layer_properties['box_id']:
            if not box_id in self.annotations:
                self.annotations[int(box_id)] = self.default_value
        
        # Switch to annotation screen
        self.stacked_widget.setCurrentIndex(1)
        
        # Initialize annotation state
        self.current_idx = 0
        self.update_display()
    
    def update_display(self):
        """Update displayed bounding box"""
        x1, y1, x2, y2 = self.layer_data[self.current_idx]
        
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
            value = values[self.current_idx]
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.props_table.item(i, 0).setToolTip(str(prop))
            self.props_table.item(i, 1).setToolTip(str(value))
        
        box_id = self.layer_properties['box_id'][self.current_idx]
        
        # Set current annotation
        current_annot = self.annotations.get(str(box_id), self.default_value)
        self.annotation_edit.setText(str(current_annot))
        self.annotation_edit.setFocus()
        self.next_btn.setEnabled(self.current_idx < len(self.layer_data) - 1)
    
    def save_annotation(self):
        """Save current annotation to dictionary"""
        box_id = self.layer_properties['box_id'][self.current_idx]
        annotation = self.annotation_edit.text()
        self.annotations[str(box_id)] = annotation
        
        annotation_features = session.SESSION_VARS.get('annotation_features', {})
        annotation_features[self.annotation_name] = {
            'name': self.property_name,
            'type': "Text",
            'data': self.annotations
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
        if self.current_idx < len(self.layer_data) - 1:
            self.current_idx += 1
            self.update_display()
    
    def get_annotations(self):
        """Return the updated annotations dictionary"""
        return self.annotations
