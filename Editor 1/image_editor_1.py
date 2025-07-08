import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QGroupBox, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage


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


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor")
        self.setFixedSize(1200, 800)
        self.setAcceptDrops(True)

        self.original_img = None
        self.current_img = None
        self.undo_stack = []
        self.redo_stack = []

        self.original_label = QLabel()
        self.filtered_label = QLabel()
        for lbl in (self.original_label, self.filtered_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setScaledContents(False)  # Maintain aspect ratio

        # Containers for side-by-side layout
        self.original_container = QWidget()
        self.filtered_container = QWidget()
        self.original_container.setLayout(QVBoxLayout())
        self.filtered_container.setLayout(QVBoxLayout())
        self.original_container.layout().addWidget(self.original_label)
        self.filtered_container.layout().addWidget(self.filtered_label)

        # Set container size policies
        self.original_container.setMinimumWidth(600)
        self.filtered_container.setMinimumWidth(600)

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

        # Sliders
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

        sliders_group = QGroupBox("Adjustments")
        sliders_group.setLayout(sliders_layout)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_container)
        image_layout.addWidget(self.filtered_container)

        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(sliders_group)

        self.setLayout(main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_display()

    def refresh_display(self):
        if self.original_img is not None and self.current_img is not None:
            self.display_images(self.original_img, self.current_img)

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
                    processed = apply_sepia(img)  # Customize filter
                    out_path = os.path.join(out_folder, "processed_" + file)
                    cv2.imwrite(out_path, processed)
            QMessageBox.information(self, "Done", f"Processed images saved in: {out_folder}")

    def reset_sliders(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)

    def display_images(self, original, filtered):
        orig_qimage = cv_to_qimage(original)
        filt_qimage = cv_to_qimage(filtered)

        orig_pix = QPixmap.fromImage(orig_qimage).scaled(
            self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        filt_pix = QPixmap.fromImage(filt_qimage).scaled(
            self.filtered_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.original_label.setPixmap(orig_pix)
        self.filtered_label.setPixmap(filt_pix)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
