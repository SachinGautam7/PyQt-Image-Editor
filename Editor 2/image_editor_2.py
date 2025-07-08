import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QGroupBox, QMessageBox, QToolBar, QAction
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QCursor


# ----------------------------
# Dark & Light Stylesheets
# ----------------------------

dark_stylesheet = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
}
QPushButton {
    background-color: #3c3f41;
    color: #f0f0f0;
    border: 1px solid #555;
    padding: 5px;
}
QPushButton:hover {
    background-color: #4c5052;
}
QSlider::groove:horizontal {
    height: 8px;
    background: #555;
}
QSlider::handle:horizontal {
    background: #aaa;
    border: 1px solid #333;
    width: 14px;
    margin: -4px 0;
}
QGroupBox {
    border: 1px solid #555;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
}
QToolBar {
    background-color: #3c3f41;
}
"""

light_stylesheet = """
QWidget {
    background-color: #f0f0f0;
    color: #222;
}
QPushButton {
    background-color: #ffffff;
    color: #222;
    border: 1px solid #999;
    padding: 5px;
}
QPushButton:hover {
    background-color: #e0e0e0;
}
QSlider::groove:horizontal {
    height: 8px;
    background: #ccc;
}
QSlider::handle:horizontal {
    background: #666;
    border: 1px solid #333;
    width: 14px;
    margin: -4px 0;
}
QGroupBox {
    border: 1px solid #999;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
}
QToolBar {
    background-color: #dcdcdc;
}
"""


# ----------------------------
# Helper Functions
# ----------------------------

def cv_to_qimage(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()


def apply_sepia(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# ----------------------------
# Custom Comparison Label
# ----------------------------

class ComparisonLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pix = None
        self.filtered_pix = None
        self.slider_pos = 0.5
        self.dragging = False
        self.setCursor(Qt.SplitHCursor)

    def set_images(self, original_pixmap, filtered_pixmap):
        self.original_pix = original_pixmap
        self.filtered_pix = filtered_pixmap
        self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.original_pix and self.filtered_pix:
            orig_scaled = self.original_pix.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            filt_scaled = self.filtered_pix.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            center_x = (self.width() - orig_scaled.width()) // 2
            center_y = (self.height() - orig_scaled.height()) // 2

            # draw original full
            painter.drawPixmap(center_x, center_y, orig_scaled)

            # overlay filtered partially
            clip_width = int(filt_scaled.width() * self.slider_pos)
            clip_rect = QRect(center_x, center_y, clip_width, filt_scaled.height())
            if clip_rect.width() > 0:
                painter.setClipRect(clip_rect)
                painter.drawPixmap(center_x, center_y, filt_scaled)

            # draw the draggable handle
            handle_x = center_x + clip_width
            pen = QPen(QColor("#ffcc00"), 3)
            painter.setPen(pen)
            painter.setClipping(False)
            painter.drawLine(handle_x, center_y, handle_x, center_y + orig_scaled.height())

            # draw handle rectangle for better look
            handle_rect = QRect(handle_x - 5, center_y + orig_scaled.height()//2 - 15, 10, 30)
            painter.fillRect(handle_rect, QColor("#ffcc00"))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.update_slider(event.pos().x())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.update_slider(event.pos().x())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def update_slider(self, x_pos):
        if self.original_pix:
            orig_scaled = self.original_pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            center_x = (self.width() - orig_scaled.width()) // 2
            left = center_x
            right = center_x + orig_scaled.width()
            x_pos = np.clip(x_pos, left, right)
            fraction = (x_pos - left) / (right - left)
            self.slider_pos = fraction
            self.repaint()


# ----------------------------
# Main App Class
# ----------------------------

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self.original_img = None
        self.current_img = None
        self.undo_stack = []
        self.redo_stack = []

        # Toolbar
        toolbar = QToolBar()
        self.dark_theme_action = QAction("Dark Theme", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.triggered.connect(lambda: self.apply_theme("dark"))
        toolbar.addAction(self.dark_theme_action)

        self.light_theme_action = QAction("Light Theme", self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.triggered.connect(lambda: self.apply_theme("light"))
        toolbar.addAction(self.light_theme_action)

        # Comparison Label
        self.comparison_label = ComparisonLabel()
        self.comparison_label.setAlignment(Qt.AlignCenter)

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.refresh_display)

        # Buttons
        load_btn = QPushButton("Load")
        save_btn = QPushButton("Save")
        batch_btn = QPushButton("Batch Process")
        gray_btn = QPushButton("Grayscale")
        sepia_btn = QPushButton("Sepia")
        edge_btn = QPushButton("Edge")
        blur_btn = QPushButton("Blur")
        sharp_btn = QPushButton("Sharpen")
        hist_btn = QPushButton("Equalize Hist.")
        undo_btn = QPushButton("Undo")
        redo_btn = QPushButton("Redo")
        reset_btn = QPushButton("Reset")

        load_btn.clicked.connect(self.load_image)
        save_btn.clicked.connect(self.save_image)
        batch_btn.clicked.connect(self.batch_process_folder)
        gray_btn.clicked.connect(self.apply_grayscale)
        sepia_btn.clicked.connect(self.apply_sepia)
        edge_btn.clicked.connect(self.apply_edge)
        blur_btn.clicked.connect(self.apply_blur)
        sharp_btn.clicked.connect(self.apply_sharpen)
        hist_btn.clicked.connect(self.apply_hist_eq)
        undo_btn.clicked.connect(self.undo)
        redo_btn.clicked.connect(self.redo)
        reset_btn.clicked.connect(self.reset_image)

        # Brightness / Contrast sliders
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_brightness_contrast)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.update_brightness_contrast)

        # Layouts
        btn_layout = QHBoxLayout()
        for btn in [load_btn, save_btn, batch_btn, gray_btn, sepia_btn, edge_btn,
                    blur_btn, sharp_btn, hist_btn, undo_btn, redo_btn, reset_btn]:
            btn_layout.addWidget(btn)

        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(QLabel("Brightness"))
        sliders_layout.addWidget(self.brightness_slider)
        sliders_layout.addWidget(QLabel("Contrast"))
        sliders_layout.addWidget(self.contrast_slider)
        sliders_layout.addWidget(QLabel("Zoom"))
        sliders_layout.addWidget(self.zoom_slider)

        sliders_group = QGroupBox("Adjustments")
        sliders_group.setLayout(sliders_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.comparison_label, stretch=1)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(sliders_group)

        self.setLayout(main_layout)

        # Apply default theme
        self.apply_theme("dark")

    def apply_theme(self, theme_name):
        if theme_name == "dark":
            self.setStyleSheet(dark_stylesheet)
            self.dark_theme_action.setChecked(True)
            self.light_theme_action.setChecked(False)
        else:
            self.setStyleSheet(light_stylesheet)
            self.dark_theme_action.setChecked(False)
            self.light_theme_action.setChecked(True)

    def load_image(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.original_img = cv2.imread(path)
            self.current_img = self.original_img.copy()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.reset_sliders()
            self.refresh_display()

    def save_image(self):
        if self.current_img is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp)")
            if path:
                cv2.imwrite(path, self.current_img)

    def batch_process_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            out_folder = os.path.join(folder, "processed")
            os.makedirs(out_folder, exist_ok=True)
            for file in os.listdir(folder):
                if file.lower().endswith(('.jpg', '.png', '.bmp')):
                    img_path = os.path.join(folder, file)
                    img = cv2.imread(img_path)
                    processed = apply_sepia(img)
                    out_path = os.path.join(out_folder, "processed_" + file)
                    cv2.imwrite(out_path, processed)
            QMessageBox.information(self, "Done", f"Processed images saved in: {out_folder}")

    def reset_sliders(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.zoom_slider.setValue(100)

    def refresh_display(self):
        if self.original_img is None or self.current_img is None:
            return

        zoom_factor = self.zoom_slider.value() / 100.0
        orig_img = self.scale_image(self.original_img, zoom_factor)
        filt_img = self.scale_image(self.current_img, zoom_factor)

        orig_q = cv_to_qimage(orig_img)
        filt_q = cv_to_qimage(filt_img)

        orig_pix = QPixmap.fromImage(orig_q)
        filt_pix = QPixmap.fromImage(filt_q)

        self.comparison_label.set_images(orig_pix, filt_pix)

    def scale_image(self, img, scale):
        height, width = img.shape[:2]
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def push_undo(self):
        if self.current_img is not None:
            self.undo_stack.append(self.current_img.copy())
            self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.current_img.copy())
            self.current_img = self.undo_stack.pop()
            self.refresh_display()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.current_img.copy())
            self.current_img = self.redo_stack.pop()
            self.refresh_display()

    def update_brightness_contrast(self):
        if self.original_img is None:
            return
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        img = self.original_img.copy()
        alpha = 1.0 + contrast / 100.0
        beta = brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        self.current_img = img
        self.refresh_display()

    def apply_grayscale(self):
        if self.current_img is not None:
            self.push_undo()
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            self.current_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.refresh_display()

    def apply_sepia(self):
        if self.current_img is not None:
            self.push_undo()
            self.current_img = apply_sepia(self.current_img)
            self.refresh_display()

    def apply_edge(self):
        if self.current_img is not None:
            self.push_undo()
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.current_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.refresh_display()

    def apply_blur(self):
        if self.current_img is not None:
            self.push_undo()
            self.current_img = cv2.GaussianBlur(self.current_img, (9, 9), 0)
            self.refresh_display()

    def apply_sharpen(self):
        if self.current_img is not None:
            self.push_undo()
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            self.current_img = cv2.filter2D(self.current_img, -1, kernel)
            self.refresh_display()

    def apply_hist_eq(self):
        if self.current_img is not None:
            self.push_undo()
            self.current_img = histogram_equalization(self.current_img)
            self.refresh_display()

    def reset_image(self):
        if self.original_img is not None:
            self.push_undo()
            self.current_img = self.original_img.copy()
            self.reset_sliders()
            self.refresh_display()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.load_image(path)
                break


# ----------------------------
# Run the Application
# ----------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
