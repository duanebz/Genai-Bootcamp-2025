import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Simple UI Demo")
        self.resize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        
        # Add a label
        label = QLabel("Hello, World!")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, 0, 0)
        
        # Add a button
        button = QPushButton("Click Me")
        button.clicked.connect(lambda: label.setText("Button Clicked!"))
        layout.addWidget(button, 1, 0)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
