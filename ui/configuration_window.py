import joblib
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QHBoxLayout, QSizePolicy, QComboBox, QToolBar,
    QAction, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QValidator, QIntValidator, QDoubleValidator, QIcon
)
from src.perceptron_model import PerceptronModel
from ui.results_window import ResultsWindow
from util.visualize_data import (
    build_graphs_for_classifier_model,
    build_graphs_for_regressor_model,
    build_graphs_for_regressor_multistage_model
)


class ConfigurationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.results_window = None
        self.inputFields = {}
        self.model = None
        self.initUI()
        self.loadDefaultModel()

        self.isDefaulted = False

    def initUI(self):
        layout = QVBoxLayout()

        # Toolbar
        self.toolbar = QToolBar("Model Tools")
        self.addToolbarActions()
        layout.addWidget(self.toolbar)

        # Parameters for the dataset
        self.addInputFieldWithTooltip('Dataset size/Time (for regressor model):', 'Number of samples in the dataset.',
                                      layout,
                                      QIntValidator(1, 1000000))
        self.addInputFieldWithTooltip('Min humidity:', 'Minimum humidity (in %) value measured by the sensor.', layout,
                                      QDoubleValidator(0.0, 100.0, 2))
        self.addInputFieldWithTooltip('Max humidity:', 'Maximum humidity (in %) value measured by the sensor.', layout,
                                      QDoubleValidator(0.0, 100.0, 2))
        self.addInputFieldWithTooltip('Min temperature:', 'Minimum temperature value measured by the sensor.', layout,
                                      QDoubleValidator(-273.15, 1000.0, 2))
        self.addInputFieldWithTooltip('Max temperature:', 'Maximum temperature value measured by the sensor.', layout,
                                      QDoubleValidator(-273.15, 1000.0, 2))

        self.addInputFieldWithTooltip('Reference humidity:', 'Reference humidity which the model will strive for.',
                                      layout,
                                      QDoubleValidator(0.0, 100.0, 2))
        self.addInputFieldWithTooltip('Reference temperature:',
                                      'Reference temperature which the model will strive for.', layout,
                                      QDoubleValidator(-273.15, 1000.0, 2))

        self.addInputFieldWithTooltip('Current humidity:',
                                      'Only for the regressor model. Current humidity in the camera.',
                                      layout,
                                      QDoubleValidator(0.0, 100.0, 2))
        self.addInputFieldWithTooltip('Current temperature:',
                                      'Only for the regressor model. Current temperature of the heater.', layout,
                                      QDoubleValidator(-273.15, 1000.0, 2))

        # Parameters for the perceptron
        layout.addWidget(QLabel('<b>Perceptron Parameters:</b>'))
        self.addInputFieldWithCustomValidation('Hidden layer sizes:', 'The size of the hidden layers (e.g., 10,10 for '
                                                                      'two layers with 10 neurons each).', layout,
                                               self.validateHiddenLayerSizes)
        activation_function_description = (
            'Activation function for the hidden layer.\n'
            'relu: Rectified Linear Unit, effective for non-linear transformations.\n'
            'tanh: Hyperbolic tangent, outputs values between -1 and 1.\n'
            'logistic: Logistic sigmoid function, outputs values between 0 and 1.'
        )
        self.addComboBoxWithTooltip('Activation function:', activation_function_description,
                                    ['relu', 'tanh', 'logistic'], layout)
        solver_description = (
            'The solver for weight optimization.\n'
            'adam: An algorithm for first-order gradient-based optimization of stochastic objective functions.\n'
            'sgd: Stochastic Gradient Descent, a simple yet very efficient approach to fitting linear classifiers and '
            'regressors under convex loss functions such as (linear) Support Vector Machines and Logistic '
            'Regression.\n '
            'lbfgs: Limited-memory Broyden-Fletcher-Goldfarb-Shanno Algorithm, an optimization algorithm in the '
            'family of quasi-Newton methods that approximates the Broyden-Fletcher-Goldfarb-Shanno algorithm using a '
            'limited amount of computer memory. '
        )
        self.addComboBoxWithTooltip('Solver:', solver_description, ['adam', 'sgd', 'lbfgs'], layout)
        self.addInputFieldWithTooltip('Alpha:', 'L2 penalty (regularization term) parameter.', layout,
                                      QDoubleValidator(0.0, 1.0, 5))
        learning_rate_description = (
            'Learning rate schedule for weight updates.\n'
            'constant: Keeps the learning rate constant throughout the training process.\n'
            'invscaling: Gradually decreases the learning rate at each step using an inverse scaling exponent of '
            'power_t.\n '
            'adaptive: Keeps the learning rate constant as long as training loss keeps decreasing. Each time two '
            'consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score '
            'by at least tol if `early_stopping` is True, the current learning rate is divided by 5. '
        )
        self.addComboBoxWithTooltip('Learning rate:', learning_rate_description, ['constant', 'invscaling', 'adaptive'],
                                    layout)
        self.addInputFieldWithTooltip('Max iter:', 'Maximum number of iterations.', layout, QIntValidator(1, 1000000))
        self.addInputFieldWithTooltip('Random seed:', 'Seed of the pseudo random number generator.', layout,
                                      QIntValidator(0, 2147483647))

        buttonLayout = QHBoxLayout()

        self.setDefaultButton = QPushButton('Set to Defaults', self)
        self.setDefaultButton.clicked.connect(self.setDefaultValues)
        buttonLayout.addWidget(self.setDefaultButton)

        self.trainClassifierModelButton = QPushButton('Train Classifier Model', self)
        self.trainClassifierModelButton.clicked.connect(self.trainClassifierModel)
        buttonLayout.addWidget(self.trainClassifierModelButton)

        self.trainRegressorModelButton = QPushButton('Train Regressor Model', self)
        self.trainRegressorModelButton.clicked.connect(self.trainRegressorModel)
        buttonLayout.addWidget(self.trainRegressorModelButton)

        self.trainRegressorMultiStageModelButton = QPushButton('Train Regressor Model (Multi Stage)', self)
        self.trainRegressorMultiStageModelButton.clicked.connect(self.trainRegressorMultiStageModel)
        buttonLayout.addWidget(self.trainRegressorMultiStageModelButton)

        layout.addLayout(buttonLayout)

        self.setLayout(layout)
        self.setWindowTitle('Model Configurations')
        self.setGeometry(100, 100, 600, 300)

    def addToolbarActions(self):
        saveAction = QAction(QIcon('icons/save.png'), 'Save Model', self)
        saveAction.triggered.connect(self.saveModel)
        self.toolbar.addAction(saveAction)

        loadAction = QAction(QIcon('icons/upload.png'), 'Load Model', self)
        loadAction.triggered.connect(self.loadModel)
        self.toolbar.addAction(loadAction)

        deleteAction = QAction(QIcon('icons/delete.png'), 'Delete Model', self)
        deleteAction.triggered.connect(self.deleteModel)
        self.toolbar.addAction(deleteAction)

    def loadDefaultModel(self):
        default_model_path = 'default_model.pkl'
        try:
            self.model = joblib.load(default_model_path)
            QMessageBox.information(self, "Model Uploaded", "Perceptron model has been successfully uploaded.")
        except FileNotFoundError:
            pass

    def saveModel(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Model Files (*.pkl)")
        if filename:
            save_model(self.model, filename)
            QMessageBox.information(self, "Model Saved", "The model has been successfully saved.")

    def loadModel(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.pkl)")
        if filename:
            self.model = load_model(filename)
            QMessageBox.information(self, "Model Loaded", "Model has been successfully loaded.")

    def deleteModel(self):
        if self.model:
            reply = QMessageBox.question(self, 'Delete Model', 'Are you sure you want to delete the current model?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.model = None
                QMessageBox.information(self, "Model Deleted", "The model has been successfully deleted.")
        else:
            QMessageBox.warning(self, "No Model", "No model currently loaded to delete.")

    def addInputFieldWithTooltip(self, label, tooltip, layout, validator=None):
        rowLayout = QHBoxLayout()

        labelWidget = QLabel(label)
        labelWidget.setAlignment(Qt.AlignLeft)

        inputField = QLineEdit(self)
        inputField.setFixedSize(250, 20)
        inputField.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        inputField.setValidator(validator)
        inputField.textChanged.connect(lambda: self.validateInput(inputField))

        questionButton = self.hintButton(tooltip)

        rowLayout.addWidget(labelWidget)
        rowLayout.addWidget(inputField)
        rowLayout.addWidget(questionButton)
        rowLayout.setStretch(1, 4)

        layout.addLayout(rowLayout)
        self.inputFields[label] = inputField

    def addInputFieldWithCustomValidation(self, label, tooltip, layout, validation_func):
        rowLayout = QHBoxLayout()
        labelWidget = QLabel(label)
        inputField = QLineEdit(self)
        inputField.setFixedSize(250, 20)
        inputField.editingFinished.connect(lambda: self.validateCustomField(inputField, validation_func))

        questionButton = self.hintButton(tooltip)

        rowLayout.addWidget(labelWidget)
        rowLayout.addWidget(inputField)
        rowLayout.addWidget(questionButton)
        layout.addLayout(rowLayout)
        self.inputFields[label] = inputField

    def addComboBoxWithTooltip(self, label, tooltip, options, layout):
        rowLayout = QHBoxLayout()
        labelWidget = QLabel(label)
        comboBox = QComboBox(self)
        comboBox.setFixedSize(250, 20)
        comboBox.addItems(options)

        questionButton = self.hintButton(tooltip)

        rowLayout.addWidget(labelWidget)
        rowLayout.addWidget(comboBox)
        rowLayout.addWidget(questionButton)
        layout.addLayout(rowLayout)
        self.inputFields[label] = comboBox

    def hintButton(self, tooltip):
        questionButton = QPushButton('?', self)
        questionButton.setToolTip(tooltip)
        questionButton.setFixedSize(22, 22)
        questionButton.setStyleSheet("QPushButton { border-radius: 11px; border : 1px solid dark-gray; }")

        return questionButton

    def validateCustomField(self, inputField, validation_func):
        if validation_func and not validation_func(inputField.text()):
            inputField.setStyleSheet("border: 1px solid red;")
            QMessageBox.warning(self, "Validation Error", "Invalid input in the field: " + inputField.text())
            return False
        else:
            inputField.setStyleSheet("")
            return True

    def validateInput(self, inputField):
        validator = inputField.validator()
        state = validator.validate(inputField.text(), 0)[0]
        if state == QValidator.Acceptable:
            inputField.setStyleSheet("")
        else:
            inputField.setStyleSheet("border: 1px solid red;")

    def validateHiddenLayerSizes(self, text: str):
        if not text:
            return False
        for num in text.split(','):
            if not num.strip().isdigit() or int(num.strip()) <= 0:
                return False
        return True

    def validateAllInputs(self):
        allValid = True
        for label, widget in self.inputFields.items():
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    widget.setStyleSheet("border: 1px solid red;")
                    allValid = False
                else:
                    widget.setStyleSheet("")
        return allValid

    def setDefaultValues(self):
        self.isDefaulted = True

        self.inputFields['Dataset size/Time (for regressor model):'].setText('100000')
        self.inputFields['Min humidity:'].setText('75,0')
        self.inputFields['Max humidity:'].setText('85,0')
        self.inputFields['Min temperature:'].setText('180,0')
        self.inputFields['Max temperature:'].setText('220,0')
        self.inputFields['Reference humidity:'].setText('80,0')
        self.inputFields['Reference temperature:'].setText('190,0')
        self.inputFields['Current humidity:'].setText('77,5')
        self.inputFields['Current temperature:'].setText('185,0')

        self.inputFields['Activation function:'].setCurrentIndex(
            self.inputFields['Activation function:'].findText('relu'))
        self.inputFields['Solver:'].setCurrentIndex(self.inputFields['Solver:'].findText('adam'))
        self.inputFields['Learning rate:'].setCurrentIndex(self.inputFields['Learning rate:'].findText('adaptive'))

        self.inputFields['Hidden layer sizes:'].setText('50, 50')
        self.inputFields['Alpha:'].setText('0,0001')
        self.inputFields['Max iter:'].setText('350')
        self.inputFields['Random seed:'].setText('0')

    def convert_to_tuple(self, text):
        try:
            return tuple(map(int, text.split(',')))
        except ValueError:
            return (10, 10)  # If smth went wrong - return default value

    def getAllInputs(self):
        if self.validateAllInputs():
            try:
                # Getting all the parameters
                self.dataset_size = int(self.inputFields['Dataset size/Time (for regressor model):'].text())
                self.min_humidity = float(str(self.inputFields['Min humidity:'].text()).replace(',', '.'))
                self.max_humidity = float(str(self.inputFields['Max humidity:'].text()).replace(',', '.'))
                self.min_temperature = float(str(self.inputFields['Min temperature:'].text()).replace(',', '.'))
                self.max_temperature = float(str(self.inputFields['Max temperature:'].text()).replace(',', '.'))
                self.ref_temp = float(str(self.inputFields['Reference temperature:'].text().replace(',', '.')))
                self.ref_humid = float(str(self.inputFields['Reference humidity:'].text().replace(',', '.')))
                self.cur_temp = float(str(self.inputFields['Current temperature:'].text().replace(',', '.')))
                self.cur_humid = float(str(self.inputFields['Current humidity:'].text().replace(',', '.')))

                self.activation_function = self.inputFields['Activation function:'].currentText()
                self.solver = self.inputFields['Solver:'].currentText()
                self.learning_rate = self.inputFields['Learning rate:'].currentText()

                self.hidden_layer_sizes = self.convert_to_tuple(self.inputFields['Hidden layer sizes:'].text())
                self.alpha = float(str(self.inputFields['Alpha:'].text()).replace(',', '.'))
                self.max_iter = int(self.inputFields['Max iter:'].text())
                self.random_state = int(self.inputFields['Random seed:'].text())

                model = PerceptronModel(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation_function,
                    solver=self.solver,
                    alpha=self.alpha,
                    learning_rate=self.learning_rate,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                return model
            except Exception as e:
                QMessageBox.critical(self, "Training Failed", f"An error occurred during training: {str(e)}")
        else:
            QMessageBox.warning(self, "Validation Error", "Please correct the highlighted fields before proceeding.")

    def trainClassifierModel(self):
        try:
            model = self.getAllInputs()
            df = model.create_dataset_for_classifier_model(
                size=self.dataset_size,
                min_temperature=self.min_temperature,
                max_temperature=self.max_temperature,
                min_humidity=self.min_humidity,
                max_humidity=self.max_humidity,
                ref_temp=self.ref_temp,
                ref_humid=self.ref_humid
            )
            X_train, X_test, y_train, y_test = model.train_test_split_for_classifier_model(df)
            X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
            classifier_model = model.train_classifier_model(X_train_scaled, y_train)
            classification_report = model.evaluate_classifier_model(classifier_model, X_test_scaled, y_test)
            with open("classifier_report.txt", "w") as f:
                f.write(f'{classification_report}')
            QMessageBox.information(self, "Training Complete",
                                    "The classifier model has been successfully trained. Report saved to classifier_report.txt")

            self.results_window = ResultsWindow(classification_report)
            self.results_window.show()

            filename1 = 'classifier_model_plots.png'
            filename2 = 'classifier_model_actions_plots.png'
            build_graphs_for_classifier_model(df=df, ref_humid=self.ref_humid, ref_temp=self.ref_temp)
            QMessageBox.information(self, "Plots Saved",
                                    f"Plots of initial data and model actions successfully saved to {filename1} and {filename2}")
        except Exception as e:
            QMessageBox.critical(self, "Training Failed", f"An error occurred during training: {str(e)}")

    def trainRegressorModel(self, initial_temp=None, initial_humid=None, ref_temp=None, ref_humid=None,
                            plot_filename='regressor_model_plot.png'):
        if self.isDefaulted:
            self.inputFields['Dataset size/Time (for regressor model):'].setText('500')
        self.isDefaulted = False

        initial_temp = initial_temp or float(str(self.inputFields['Current temperature:'].text()).replace(',', '.'))
        initial_humid = initial_humid or float(str(self.inputFields['Current humidity:'].text()).replace(',', '.'))
        ref_temp = ref_temp or float(str(self.inputFields['Reference temperature:'].text()).replace(',', '.'))
        ref_humid = ref_humid or float(str(self.inputFields['Reference humidity:'].text()).replace(',', '.'))

        try:
            self.Kp = 0.1
            self.Ki = 0.01
            self.Kd = 0.05

            model = self.getAllInputs()
            df = model.create_dataset_for_regressor_model(
                time=self.dataset_size,
                initial_temp=initial_temp,
                initial_humid=initial_humid,
                ref_temp=ref_temp,
                ref_humid=ref_humid,
                Kp=self.Kp,
                Ki=self.Ki,
                Kd=self.Kd
            )
            X_train, X_test, y_train, y_test = model.train_test_split_for_regressor_model(df)
            X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
            regressor_model = model.train_regressor_model(X_train_scaled, y_train)
            mse, mae, r2 = model.evaluate_regressor_model(regressor_model, X_test_scaled, y_test)

            QMessageBox.information(self, "Training Complete",
                                    f"The regressor model has been successfully trained. Results:\nMSE: {mse}, "
                                    f"\nMAE: {mae}\nR^2: {r2}")

            build_graphs_for_regressor_model(df=df, ref_temp=self.ref_temp, ref_humid=self.ref_humid,
                                             cur_temp=self.cur_temp, cur_humid=self.cur_humid)
            QMessageBox.information(self, "Plots Saved",
                                    f"Plots of the transition characteristics for the temperature and humidity successfuly saved to {plot_filename}")
        except Exception as e:
            QMessageBox.critical(self, "Training Failed", f"An error occurred during training: {str(e)}")

    def trainRegressorMultiStageModel(self):
        if self.isDefaulted:
            self.inputFields['Dataset size/Time (for regressor model):'].setText('500')
        self.isDefaulted = False

        try:
            self.Kp = 0.1
            self.Ki = 0.01
            self.Kd = 0.05

            model = self.getAllInputs()
            stage_params = [
                {'initial_temp': 0, 'initial_humid': 0, 'ref_temp': 120, 'ref_humid': 85,
                 'duration': self.dataset_size // 3},
                {'initial_temp': 120, 'initial_humid': 85, 'ref_temp': 250, 'ref_humid': 10,
                 'duration': self.dataset_size // 3},
                {'initial_temp': 250, 'initial_humid': 10, 'ref_temp': 180, 'ref_humid': 15,
                 'duration': self.dataset_size - 2 * (self.dataset_size // 3)}
            ]

            df = model.create_dataset_for_regressor_multistage_model(
                total_time=self.dataset_size,
                stage_params=stage_params,
                Kp=self.Kp,
                Ki=self.Ki,
                Kd=self.Kd
            )
            X_train, X_test, y_train, y_test = model.train_test_split_for_regressor_model(df)
            X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
            regressor_model = model.train_regressor_model(X_train_scaled, y_train)
            mse, mae, r2 = model.evaluate_regressor_model(regressor_model, X_test_scaled, y_test)

            QMessageBox.information(self, "Training Complete",
                                    f"The regressor model has been successfully trained. Results:\nMSE: {mse}, "
                                    f"\nMAE: {mae}\nR^2: {r2}")

            filename = 'multi_stage_regressor_model_plots.png'
            build_graphs_for_regressor_multistage_model(df, stage_params, filename_to_save=filename)
            QMessageBox.information(self, "Plots Saved",
                                    f"Plots of the transition characteristics for all stages successfully saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Training Failed", f"An error occurred during training: {str(e)}")


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)
