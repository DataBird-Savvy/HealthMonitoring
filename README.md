
---
## 📸 Screenshots
### 🟥 Real-Time Patient Monitoring Dashboard
![heathmonitor](https://github.com/user-attachments/assets/49136bed-aa50-41fc-a8e9-78f391f0bed4)



# 🏥 Patient Health Monitoring System

## 📌 Overview
This project integrates **wearable device readings (Blood Pressure, Blood Sugar, etc.)** and **lab records** to provide **real-time health monitoring**. It employs an **LSTM-based deep learning model** to predict patient vitals and generate **alerts for doctors** when abnormalities are detected.

## 🔑 Key Features
- ✅ **Wearable Data Integration** - Monitors **Blood Pressure, Blood Sugar, Heart Rate, Oxygen Saturation, etc.**
- ✅ **Lab Record Merging** - Combines periodic lab results like **Hemoglobin, Cholesterol, WBC Count, etc.**
- ✅ **LSTM-Based Prediction** - Forecasts the **next sequence** of patient vitals for early detection.
- ✅ **Automated Alerts** - Flags abnormal readings & alerts doctors with **color-coded reports**.
- ✅ **Interactive Dashboard** - Built using **Streamlit** for real-time monitoring.

## 🚀 Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/PatientHealthMonitor.git
cd PatientHealthMonitor
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Setup Environment Variables
Create a `.env` file in the root directory and add:
```env
MERGED_DATA_PATH=data/merged_data.csv
ENCODER_PATH=models/encoder.pkl
SCALER_PATH=models/scaler.pkl
MODEL_PATH=models/lstm_model.h5
```

### 5️⃣ Run the Streamlit Dashboard
```bash
streamlit run app.py
```

## 📊 LSTM Model Architecture
The model predicts the next sequence of patient vitals using **Bidirectional LSTM layers** with **dropout regularization**:
```python
model = Sequential([
    Bidirectional(LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(30, X.shape[2])),
    LayerNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(64, activation='tanh', return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(32, activation='tanh')),
    LayerNormalization(),
    Dropout(0.2),
    Dense(X.shape[2])
])

model.compile(optimizer=AdamW(learning_rate=0.001), loss='mae')
```

## 📢 Alerts & Critical Thresholds
If a patient's predicted vitals exceed predefined thresholds, the system generates alerts:
| Feature | Normal Range |
|---------|-------------|
| Blood Glucose | 70 - 140 mg/dL |
| Oxygen (SpO₂) | 90 - 100% |
| ECG | 0.5 - 1.5 |
| Heart Rate | 50 - 110 bpm |
| Respiratory Rate | 12 - 20 breaths/min |
| Body Temperature | 36.1 - 37.8°C |

## 🛠️ Technologies Used
- **Python** (Data Processing, Model Training)
- **TensorFlow / Keras** (LSTM Model)
- **Scikit-Learn** (Preprocessing & Feature Scaling)
- **Streamlit** (Dashboard for Visualization)
- **Pandas / NumPy** (Data Manipulation)
- **Joblib** (Model Serialization)

## 🤝 Contributing
Feel free to contribute! Fork the repository and submit a **pull request**.

## 📜 License
This project is **MIT Licensed**. See `LICENSE` for details.

---
✨ **Developed with ❤️ by jiya** ✨


---


### 📜 License
🔖 MIT License - Feel free to use & modify! 🚀

