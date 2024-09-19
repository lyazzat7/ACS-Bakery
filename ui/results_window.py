from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton
)


class ResultsWindow(QWidget):
    def __init__(self, classification_report):
        super().__init__()
        self.classification_report = classification_report
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Label', 'Precision', 'Recall', 'F1-score', 'Support'])
        self.populateTable(self.classification_report)

        layout.addWidget(self.table)

        closeButton = QPushButton('Close', self)
        closeButton.clicked.connect(self.close)
        layout.addWidget(closeButton)

        self.setLayout(layout)
        self.setWindowTitle('Classification Report')
        self.setGeometry(100, 100, 600, 300)

    def populateTable(self, classification_report):
        lines = classification_report.split('\n')
        import re
        report_data = []
        for line in lines[2:]:
            parts = re.split(r'\s{2,}', line.strip())
            if parts and len(parts) >= 2:
                report_data.append(parts)

        self.table.setRowCount(len(report_data))

        for i, row in enumerate(report_data):
            if "accuracy" in row:
                self.table.setItem(i, 0, QTableWidgetItem("accuracy"))
                self.table.setItem(i, 1, QTableWidgetItem(""))
                self.table.setItem(i, 2, QTableWidgetItem(""))
                self.table.setItem(i, 3, QTableWidgetItem(row[1]))
                self.table.setItem(i, 4, QTableWidgetItem(row[2]))
            if len(row) == 5:
                for j, item in enumerate(row):
                    cell = QTableWidgetItem(item)
                    self.table.setItem(i, j, cell)
            elif len(row) == 4:
                cell = QTableWidgetItem(row[0])  # Label
                self.table.setItem(i, 0, cell)
                for j in range(1, 4):
                    cell = QTableWidgetItem(row[j])
                    self.table.setItem(i, j + 1, cell)

        self.table.resizeColumnsToContents()

