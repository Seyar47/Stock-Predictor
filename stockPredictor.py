import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
                             QComboBox, QProgressBar, QTabWidget, QMessageBox,
                             QGroupBox, QSizePolicy) 
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon 
import matplotlib
matplotlib.use('Qt5Agg') 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# qss stylesheet
QSS_STYLE = """
/* Global Styles */
QMainWindow, QWidget {
    background-color: #f0f2f5; 
    font-family: "Segoe UI", Arial, Helvetica, sans-serif; 
    color: #333333; 
}

QLabel {
    font-size: 10pt;
    padding: 2px;
}

QLabel#TitleLabel {
    font-size: 18pt;
    font-weight: bold;
    color: #2c3e50; 
    padding-bottom: 10px;
    padding-top: 5px;
}

QLabel#SectionHeaderLabel {
    font-size: 12pt;
    font-weight: bold;
    color: #34495e; 
    margin-top: 10px;
    margin-bottom: 5px;
}

QLabel#CurrentPriceLabel {
    font-size: 15pt; 
    font-weight: bold;
    color: #16a085; 
    padding: 8px; 
    background-color: #e8f6f3;
    border-radius: 5px;
    border: 1px solid #d0e8e1;
}

QLineEdit, QTextEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #d1d8e0; 
    border-radius: 5px;
    padding: 6px;
    font-size: 10pt;
    color: #333; 
}

QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #3498db; 
}

QTextEdit {
    selection-background-color: #aed6f1;
    selection-color: #222;
}

QPushButton {
    background-color: #3498db; 
    color: white;
    font-size: 10pt;
    font-weight: bold;
    padding: 10px 18px;
    border-radius: 5px;
    border: none;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #2980b9; 
}

QPushButton:pressed {
    background-color: #1f618d; 
}

QPushButton:disabled {
    background-color: #bdc3c7;
    color: #7f8c8d;
}

QTabWidget::pane {
    border: 1px solid #d1d8e0;
    border-top: none;
    background: #ffffff;
    border-bottom-left-radius: 5px;
    border-bottom-right-radius: 5px;
    padding: 10px;
}

QTabBar::tab {
    background: #e4e7eb;
    color: #566573;
    border: 1px solid #d1d8e0;
    border-bottom: none; 
    padding: 10px 20px;
    margin-right: 1px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}

QTabBar::tab:selected {
    background: #ffffff; 
    color: #3498db; 
    font-weight: bold;
    border-bottom-color: #ffffff; 
}

QTabBar::tab:hover:!selected {
    background: #eff1f3;
}

QProgressBar {
    border: 1px solid #d1d8e0;
    border-radius: 5px;
    text-align: center;
    color: #333333;
    font-size: 9pt;
    height: 22px;
}

QProgressBar::chunk {
    background-color: #27ae60;
    border-radius: 4px;
    margin: 1px;
}

QGroupBox {
    font-size: 11pt;
    font-weight: bold;
    color: #34495e;
    border: 1px solid #d1d8e0;
    border-radius: 5px;
    margin-top: 10px; 
    padding-top: 20px; 
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left; 
    padding: 0 5px 0 5px;
    left: 10px; 
    background-color: #f0f2f5; 
}

QMessageBox QLabel {
    font-size: 10pt;
    color: #333;
}

QMessageBox QPushButton {
    min-width: 80px;
}
"""

class StockPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.trained_model_name_for_prediction = 'Random Forest' # defualt model for predictions

    def fetch_data(self, ticker, period="5y"):
        try:
            stock = yf.Ticker(ticker)
            self.data = stock.history(period=period)
            if self.data.empty:
                return False, f"no data found for ticker {ticker} with period {period}."
            return True, f"Successfully fetched {len(self.data)} days of data for {ticker}"
        except Exception as e:
            return False, f"Error fetching data for {ticker}: {str(e)}"

    def preprocess_data(self):
        if self.data is None or self.data.empty:
            return False, "No data available to preprocess."
        
        try:
            self.data = self.data.dropna(subset=['Close'])
            if self.data.empty:
                return False, "No valid historical data (e.g., 'Close' price missing) after initial NaN drop."
            
            self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
            self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['Prev_Close'] = self.data['Close'].shift(1)
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            self.data['Price_Change'] = self.data['Close'] - self.data['Prev_Close']
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=5).mean()
            self.data['Target'] = self.data['Close'].shift(-1)
            self.data = self.data.dropna()
            
            if self.data.empty:
                return False, "Not enough data to create features (e.g., too short history for MAs or target)."
            
            return True, "Data preprocessing completed successfully"
        except Exception as e:
            return False, f"Error in preprocessing: {str(e)}"

    def prepare_features(self):
        feature_columns = ['Close', 'MA_5', 'MA_10', 'MA_20', 'Prev_Close', 
                           'Daily_Return', 'Price_Change', 'Volume', 'Volume_MA']
        X = self.data[feature_columns].values
        y = self.data['Target'].values
        return X, y

    def train_models(self):
        if self.data is None or self.data.empty:
            return None, None, None, "Cannot train models: No preprocessed data available."

        X, y = self.prepare_features()
        if X.shape[0] < 2: # we need at least 2 samples to split
             return None, None, None, "Cannot train models: Not enough data after feature preparation."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if len(X_train) == 0 or len(X_test) == 0 :
            return None, None, None, "Cannot train models: Data split resulted in empty train/test set."

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_init = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        results = {}
        self.models = {} # previous models needs to be cleared
        
        for name, model_instance in models_init.items():
            try:
                if name == 'Linear Regression' or name == 'KNN':
                    model_instance.fit(X_train_scaled, y_train)
                    y_pred = model_instance.predict(X_test_scaled)
                else:
                    model_instance.fit(X_train, y_train)
                    y_pred = model_instance.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {'model': model_instance, 'mse': mse, 'r2': r2, 'predictions': y_pred, 'actual': y_test}
                self.models[name] = model_instance # store the trained model
            except Exception as e:
                results[name] = {'model': None, 'mse': float('inf'), 'r2': float('-inf'), 'predictions': None, 'actual': y_test, 'error': str(e)}
        
        # Determine best model (e.g. by R2 score) for future predictions if not set or default preferred
        self.trained_model_name_for_prediction = 'Random Forest' 
        if 'Random Forest' not in self.models or self.models['Random Forest'] is None:
            # Fallback if RF failed
            best_model_name = None
            best_r2 = -float('inf')
            for name, res in results.items():
                if res['model'] is not None and res['r2'] > best_r2:
                    best_r2 = res['r2']
                    best_model_name = name
            if best_model_name:
                 self.trained_model_name_for_prediction = best_model_name

        return results, X_test, y_test, "Model training completed."

    def predict_future(self, days=7, model_name=None):
        if model_name is None:
            model_name = self.trained_model_name_for_prediction

        if model_name not in self.models or self.models[model_name] is None:
            return None, f"Model '{model_name}' not found or not trained successfully."
        
        model = self.models[model_name]
        
        feature_columns = ['Close', 'MA_5', 'MA_10', 'MA_20', 'Prev_Close', 
                           'Daily_Return', 'Price_Change', 'Volume', 'Volume_MA']
        
        if self.data is None or self.data.empty:
            return None, "No data available for prediction base."

        try:
            last_features_df = self.data[feature_columns].iloc[-1:].copy()
        except IndexError:
             return None, "Not enough historical data for prediction base."

        predictions = []
        current_features_np = last_features_df.values.copy()
        
        idx_close = feature_columns.index('Close')
        idx_prev_close = feature_columns.index('Prev_Close')
        idx_daily_return = feature_columns.index('Daily_Return')
        idx_price_change = feature_columns.index('Price_Change')

        for _ in range(days):
            if current_features_np.ndim == 1:
                current_features_np = current_features_np.reshape(1, -1)

            if model_name == 'Linear Regression' or model_name == 'KNN':
                scaled_features = self.scaler.transform(current_features_np)
                pred = model.predict(scaled_features)[0]
            else:
                pred = model.predict(current_features_np)[0]
            predictions.append(pred)
            
            close_of_day_d = current_features_np[0, idx_close]
            current_features_np[0, idx_close] = pred
            current_features_np[0, idx_prev_close] = close_of_day_d
            
            if close_of_day_d != 0:
                current_features_np[0, idx_daily_return] = (pred - close_of_day_d) / close_of_day_d
            else:
                current_features_np[0, idx_daily_return] = 0.0
            current_features_np[0, idx_price_change] = pred - close_of_day_d
            
        return predictions, "Predictions generated successfully"

    def get_current_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            current_data = stock.history(period="1d")
            if not current_data.empty:
                return current_data['Close'].iloc[-1]
            else:
                hist_short = stock.history(period="5d") 
                if not hist_short.empty:
                    return hist_short['Close'].iloc[-1]
                return None 
        except Exception as e:
            print(f"Error in get_current_price for {ticker}: {e}") 
            return None

class DataWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
        self.predictor = StockPredictor()
    
    def run(self):
        try:
            self.progress.emit(f"Fetching stock data for {self.ticker}...")
            success, msg = self.predictor.fetch_data(self.ticker)
            if not success:
                self.error.emit(msg); return
            
            self.progress.emit("Preprocessing data...")
            success, msg = self.predictor.preprocess_data()
            if not success:
                self.error.emit(msg); return
            
            self.progress.emit("Training models...")
            results, _, _, train_msg = self.predictor.train_models() # X_test, y_test not used directly by GUI here
            self.progress.emit(train_msg) 
            if results is None: # big big failure in training
                self.error.emit(train_msg if train_msg else "Model training failed critically."); return

            self.progress.emit("Generating predictions...")
            # use the model determined as best or default during training
            predictions, pred_msg = self.predictor.predict_future(model_name=self.predictor.trained_model_name_for_prediction)
            if predictions is None:
                self.error.emit(pred_msg); return
            
            current_price = self.predictor.get_current_price(self.ticker)
            
            result_data = {
                'predictor': self.predictor, 'results': results, 'predictions': predictions,
                'current_price': current_price, 'ticker': self.ticker,
                'prediction_model_name': self.predictor.trained_model_name_for_prediction
            }
            self.finished.emit(result_data)
        except Exception as e:
            self.error.emit(f"Critical error in processing thread: {str(e)}")

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        try:
            plt.style.use("seaborn-v0_8-pastel") 
        except IOError:
            print("Matplotlib style 'seaborn-v0_8-pastel' not found. Using default.")
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("white")
            
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _setup_ax_style(self, ax, title):
        ax.set_title(title, fontsize=14, fontweight='bold', color="#2c3e50")
        ax.title.set_position([.5, 1.05]) 
        ax.set_xlabel(ax.get_xlabel(), fontsize=11, color="#444")
        ax.set_ylabel(ax.get_ylabel(), fontsize=11, color="#444")
        ax.tick_params(axis='x', colors='#555', labelsize=9)
        ax.tick_params(axis='y', colors='#555', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
        if ax.legend_ != None: 
            ax.legend(fontsize=9, frameon=True, facecolor='#fdfefe', edgecolor='#d1d8e0')


    def plot_stock_history(self, data, ticker):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(data.index, data['Close'], label='Close Price', linewidth=1.5, color="#007ACC")
        ax.plot(data.index, data['MA_5'], label='5-day MA', alpha=0.7, linestyle='--', color="#F08A5D")
        ax.plot(data.index, data['MA_10'], label='10-day MA', alpha=0.7, linestyle='--', color="#B83B5E")
        ax.plot(data.index, data['MA_20'], label='20-day MA', alpha=0.7, linestyle='--', color="#6A2C70")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        self._setup_ax_style(ax, f'{ticker} Stock Price History')
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_predictions(self, actual, predicted, model_name):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.scatter(actual, predicted, alpha=0.6, edgecolors='#555', color="#5DADE2", s=30)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=1.5, label="Ideal Fit")
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        self._setup_ax_style(ax, f'{model_name}: Actual vs Predicted')
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_future_predictions(self, predictions, current_price, ticker):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        days_axis = range(1, len(predictions) + 1)

        plot_label = "Predicted Prices"
        line_color = "#FF6F61" 
        marker_color = "#FF6F61"

        if current_price is not None:
            ax.plot([0] + list(days_axis), [current_price] + predictions, 'o-', linewidth=2, markersize=6, label=plot_label, color=line_color, markerfacecolor=marker_color)
            ax.axhline(y=current_price, color='#27AE60', linestyle=':', alpha=0.8, label=f'Current: ${current_price:.2f}')
            ax.annotate(f'${current_price:.2f}', (0, current_price), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color="green", weight='bold')
        else:
            ax.plot(list(days_axis), predictions, 'o-', linewidth=2, markersize=6, label=plot_label, color=line_color, markerfacecolor=marker_color)
            ax.text(0.05, 0.95, "Current Price: N/A", transform=ax.transAxes, fontsize=10, va='top', color='gray')

        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('Predicted Price ($)')
        self._setup_ax_style(ax, f'{ticker} - {len(predictions)}-Day Price Forecast')
        
        for i, pred_val in enumerate(predictions):
            day_x = days_axis[i]
            ax.annotate(f'${pred_val:.2f}', (day_x, pred_val), textcoords="offset points", 
                        xytext=(0, -15 if i % 2 == 0 else 10), ha='center', fontsize=8, color="#4A4A4A",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f2f5", ec="none", alpha=0.7)) 
        
        self.fig.tight_layout(pad=2.0)
        self.draw()

class StockPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.results = None
        self.last_analysis_data = None 
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🔮 Stock Price Predictor Pro')
        self.setGeometry(50, 50, 1280, 850) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15) 
        main_layout.setSpacing(10) 

        title = QLabel('🔮 Stock Price Predictor Pro')
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        input_group_box = QGroupBox("Stock Analyzer")
        input_group_layout = QHBoxLayout()
        input_group_layout.addWidget(QLabel('Ticker:'))
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText('e.g., AAPL, TSLA, MSFT')
        input_group_layout.addWidget(self.ticker_input, 1)
        self.analyze_btn = QPushButton('🚀 Analyze')
        self.analyze_btn.clicked.connect(self.analyze_stock)
        input_group_layout.addWidget(self.analyze_btn)
        input_group_box.setLayout(input_group_layout)
        main_layout.addWidget(input_group_box)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel('Enter a stock ticker and click Analyze.')
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.results_tab = QWidget()
        self.tab_widget.addTab(self.results_tab, '📊 Results & Predictions')
        
        self.charts_tab = QWidget()
        self.tab_widget.addTab(self.charts_tab, '📈 Charts')
        
        self.setup_results_tab()
        self.setup_charts_tab()

    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        layout.setSpacing(15)

        # current price Section
        price_group = QGroupBox("Real-time Data")
        price_layout = QVBoxLayout()
        self.current_price_label = QLabel('Current Price: N/A')
        self.current_price_label.setObjectName("CurrentPriceLabel")
        self.current_price_label.setAlignment(Qt.AlignCenter)
        price_layout.addWidget(self.current_price_label)
        price_group.setLayout(price_layout)
        layout.addWidget(price_group)

        #  performance model Section
        perf_group = QGroupBox("Model Performance")
        perf_layout = QVBoxLayout()
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.performance_text.setFixedHeight(180) 
        perf_layout.addWidget(self.performance_text)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # predictions Section
        pred_group = QGroupBox("Price Forecast (7-Day)")
        pred_layout = QVBoxLayout()
        self.predictions_text = QTextEdit()
        self.predictions_text.setReadOnly(True)
        self.predictions_text.setFixedHeight(180) 
        pred_layout.addWidget(self.predictions_text)
        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)

        layout.addStretch(1) 

    def setup_charts_tab(self):
        layout = QVBoxLayout(self.charts_tab)
        layout.setSpacing(10)
        
        chart_selection_layout = QHBoxLayout()
        chart_selection_layout.addWidget(QLabel('Select Chart:'))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(['Stock History', 'Model Performance (Random Forest)', 'Future Price Forecast'])
        self.chart_combo.currentTextChanged.connect(self.update_chart)
        chart_selection_layout.addWidget(self.chart_combo, 1)
        layout.addLayout(chart_selection_layout)
        
        self.plot_canvas = PlotCanvas(self)
        layout.addWidget(self.plot_canvas)

    def analyze_stock(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, 'Input Required', 'Please enter a stock ticker symbol.')
            return
        
        self.analyze_btn.setEnabled(False)
        self.status_label.setText(f"Starting analysis for {ticker}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.worker = DataWorker(ticker)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def on_analysis_complete(self, data):
        self.last_analysis_data = data
        self.predictor = data['predictor']
        self.results = data['results']
        current_price = data['current_price']
        predictions = data['predictions']
        ticker = data['ticker']
        pred_model_name = data['prediction_model_name']

        if current_price is not None:
            self.current_price_label.setText(f'Current Price ({ticker}): ${current_price:.2f}')
        else:
            self.current_price_label.setText(f'Current Price ({ticker}): N/A')
        
        perf_text = ""
        if self.results:
            for name, result in self.results.items():
                if result.get('error'):
                    perf_text += f"{name}:\n  Error: {result['error']}\n\n"
                else:
                    perf_text += f"{name}:\n"
                    perf_text += f"  Mean Squared Error: {result['mse']:.4f}\n"
                    perf_text += f"  R² Score: {result['r2']:.4f}\n\n"
        else:
            perf_text = "Model performance data not available."
        self.performance_text.setText(perf_text.strip())
        
        if predictions:
            pred_text = f"Forecast using {pred_model_name} model:\n\n"
            for i, pred in enumerate(predictions, 1):
                pred_text += f"Day {i}: ${pred:.2f}\n"
        else:
            pred_text = "Price forecast not available."
        self.predictions_text.setText(pred_text.strip())
        
        self.update_chart()
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0,100)
        self.status_label.setText(f'Analysis complete for {ticker}. Ready for new analysis or chart selection.')
        QMessageBox.information(self, "Analysis Complete", f"Successfully analyzed {ticker}.")


    def on_analysis_error(self, error_msg):
        QMessageBox.critical(self, 'Analysis Error', error_msg)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText('Analysis failed. Please try again or check ticker.')

    def update_chart(self):
        if not self.last_analysis_data or not self.predictor or not self.results:
            self.plot_canvas.fig.clear()
            ax = self.plot_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data to display.\nPlease analyze a stock first.", 
                      ha='center', va='center', fontsize=12, color="#777")
            self.plot_canvas.draw()
            return
        
        chart_type = self.chart_combo.currentText()
        ticker = self.last_analysis_data['ticker']
        
        self.plot_canvas.fig.clear() 

        if chart_type == 'Stock History':
            if self.predictor.data is not None and not self.predictor.data.empty:
                self.plot_canvas.plot_stock_history(self.predictor.data, ticker)
            else:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No historical data to display.", ha='center', va='center')
                self.plot_canvas.draw()
        elif chart_type == 'Model Performance (Random Forest)': 
            model_key = 'Random Forest'
            if model_key in self.results and \
               isinstance(self.results[model_key].get('actual'), np.ndarray) and \
               isinstance(self.results[model_key].get('predictions'), np.ndarray):
                rf_result = self.results[model_key]
                self.plot_canvas.plot_predictions(rf_result['actual'], rf_result['predictions'], model_key)
            else:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, f"{model_key} performance data not available.", ha='center', va='center')
                self.plot_canvas.draw()
        elif chart_type == 'Future Price Forecast':
            predictions = self.last_analysis_data['predictions']
            current_price = self.last_analysis_data['current_price']
            if predictions is not None:
                self.plot_canvas.plot_future_predictions(predictions, current_price, ticker)
            else:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "Future price forecast not available.", ha='center', va='center')
                self.plot_canvas.draw()
        else:
            self.plot_canvas.draw() 

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS_STYLE) 
    
    try:
        app.setWindowIcon(QIcon('stock.png')) 
    except Exception as e:
        print(f"Could not load app icon: {e}")

    window = StockPredictorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()