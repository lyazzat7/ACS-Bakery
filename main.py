import sys
from ui.configuration_window import ConfigurationWindow
from util.dependecy_installer import DependencyInstaller
from PyQt5.QtWidgets import QApplication

DependencyInstaller().install_dependencies(
    ["pandas", "scikit-learn", "numpy", "PyQt5",
     "joblib", "pydot", "matplotlib"])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ConfigurationWindow()
    ex.show()
    sys.exit(app.exec_())
