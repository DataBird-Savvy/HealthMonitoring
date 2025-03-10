# 🏥 Health Monitoring System - AI-Powered Patient Alerts

### 🚀 Real-Time Patient Monitoring using Wearable Data, Lab Records & LSTM Prediction

---

## 📌 Overview
This project integrates **wearable device readings (Blood Pressure, Blood Sugar)** and **lab records** to provide **real-time health monitoring**. It employs an **LSTM model** to predict the next sequence of patient vitals, generating **alerts for doctors** when abnormalities are detected.  

### 🔑 Key Features
- ✅ **Wearable Data Integration** - Monitors **BP, Blood Sugar, Heart Rate, Oxygen Saturation, etc.**  
- ✅ **Lab Record Merging** - Combines periodic lab results like **Hemoglobin, Cholesterol, WBC Count, etc.**  
- ✅ **LSTM-Based Prediction** - Forecasts the **next sequence** of patient vitals for early detection  
- ✅ **Automated Alerts** - Flags abnormal readings & alerts doctors with **color-coded reports**  
- ✅ **Streamlit Dashboard** - Live monitoring with **auto-refresh every 5 minutes**  

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, TensorFlow/Keras, Scikit-learn)  
- **Streamlit** (Real-time dashboard for patient monitoring)  
- **LSTM (Long Short-Term Memory)** - Time-series prediction model  
- **MongoDB** (For storing patient records)  
- **Matplotlib/Seaborn** (Data visualization)  

---

## 📂 Data Sources
- **Wearable Device Data**: Continuous BP, Glucose, HR, SpO₂  
- **Lab Records**: Periodic lab test reports (Hemoglobin, WBC Count, etc.)  
- **Historical Patient Data**: Used for LSTM training  

---

## 📊 Model Training - LSTM
The **LSTM model** predicts the **next sequence** of patient vitals based on historical data.  
### Training Steps:
1. **Data Preprocessing** - Filling missing values, normalizing sensor data  
2. **Sequence Generation** - Creating time-series input for LSTM  
3. **Model Training** - Using **TensorFlow/Keras LSTM**  
4. **Prediction** - Generating future vitals & checking alert thresholds  

---

## 🚨 Alert System - Thresholds
The system automatically triggers **alerts for doctors** if values go **beyond normal thresholds**:

| Parameter              | Normal Range |
|------------------------|--------------|
| Systolic BP           | 90 - 120 mmHg |
| Diastolic BP          | 60 - 80 mmHg  |
| Blood Sugar (Glucose) | 70 - 140 mg/dL |
| Oxygen Saturation (SpO₂) | 95 - 100% |
| Heart Rate            | 60 - 100 bpm |
| Temperature           | 36.1 - 37.5°C |
| Respiratory Rate      | 12 - 20 breaths/min |
| Electrocardiogram (ECG) | 0.88 - 1.02 |
| Hydration Level       | 55 - 65% |

🛥️ **Red Highlight** = Critical Alert  
🟩 **Green Highlight** = Normal  

---

## 📸 Screenshots
### 🟥 Real-Time Patient Monitoring Dashboard
![Patient Monitoring UI](https://your-image-link.png)

---

## 🚀 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HealthMonitoringSystem.git
   cd HealthMonitoringSystem
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Future Improvements
- 🛠️ **Real-time IoT Device Integration** for direct sensor streaming  
- ⚖️ **Edge Computing** for low-latency health monitoring  
- 📢 **Doctor-Patient Communication System** via alerts  

---


### 📜 License
🔖 MIT License - Feel free to use & modify! 🚀

