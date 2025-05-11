
---
## 📸 Screenshots
### 🟥 Real-Time Patient Monitoring Dashboard
![heathmonitor](https://github.com/user-attachments/assets/49136bed-aa50-41fc-a8e9-78f391f0bed4)



# 🏥 Patient Health Monitoring System

## 📌 Overview
This project integrates **wearable device readings (Blood Pressure, Blood Sugar, etc.)** and **lab records** to provide **real-time health monitoring**. It employs an **LSTM-based deep learning model** to predict patient vitals and generate **alerts for doctors** when abnormalities are detected. The system now supports **Kafka-based data streaming** and **dataset drift monitoring** using **Evidently AI**.

---

## 🔑 Key Features
- ✅ **Wearable Data Integration** - Monitors **Blood Pressure, Blood Sugar, Heart Rate, Oxygen Saturation, etc.**
- ✅ **Lab Record Merging** - Combines periodic lab results like **Hemoglobin, Cholesterol, WBC Count, etc.**
- ✅ **Kafka Streaming** - Real-time ingestion via **Kafka** and **Zookeeper** (via Docker Compose).
- ✅ **LSTM-Based Prediction** - Forecasts the **next sequence** of patient vitals for early detection.
- ✅ **Dataset Drift Detection** - Uses **Evidently AI** to detect **drift** in incoming data.
- ✅ **Automated Alerts** - Flags abnormal readings & alerts doctors with **color-coded reports**.
- ✅ **Interactive Dashboard** - Built using **Streamlit** for real-time monitoring.

---

## 🐳 Kafka & Zookeeper Setup (Docker Compose)
Make sure Docker is installed. Then run:
```bash
docker-compose up -d

---
✨ **Developed with ❤️ by jiya** ✨


---


### 📜 License
🔖 MIT License - Feel free to use & modify! 🚀

