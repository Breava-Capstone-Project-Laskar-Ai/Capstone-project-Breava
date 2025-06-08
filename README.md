# **Capstone Project BREAVA**

## Overview

We built machine learning using LSTM (Long Short Term Memory) model for BREAVA application using TensorFlow as library and Google Colab as tools for training.

## Dataset
We collected and chose the most completed dataset from Kaggle entitled [Air Quality in Yogyakarta, Indonesia (2021)](https://www.kaggle.com/datasets/adhang/air-quality-in-yogyakarta-indonesia-2021). 
This dataset contains air pollution measurements, such as Particulate Matter (PM10, PM2.5), Sulfur Dioxide (SO2), Carbon Monoxide (CO), Ozone (O3), and Natrium Dioxide (NO2). The measurement has been converted to pollutant standards index (PSI) or Indeks Standar Pencemaran Udara (ISPU).

Attribute Information

> Date - Date of measurements

> PM10 - Particulate Matter measurements

> PM2.5 - Particulate Matter measurements

> SO2 - Sulfur Dioxide measurements

> CO - Carbon Monoxide measurements

> O3 - Ozone measurements

> NO2 - Natrium DIoxide measurements

> Max - The highest measurement value

> Critical Component - Component(s) that has the highest measurement value

> Category - Category of air pollution, whether it's good or not

## Model
We use LSTM for prediction
These are the model result:

![image](https://github.com/user-attachments/assets/ee4ea73f-a5a7-4b90-95dc-66b6afb637aa)
![image](https://github.com/user-attachments/assets/85de419b-2b3a-42d8-be58-dc63e4b376ee)
![image](https://github.com/user-attachments/assets/b866b070-752a-4db6-81cd-4bd617a4ac80)

## Result App

You can access the website with this link [Breava Application](https://capstone-project-breava-cv6nhpbjqjwyuu2uryx7hz.streamlit.app/)

![image](https://github.com/user-attachments/assets/f97ecf79-dcd1-4bfb-996b-cd2e8a92c483)
![image](https://github.com/user-attachments/assets/e3960a51-1e9b-4bf2-9b9c-6b2718301c18)
![image](https://github.com/user-attachments/assets/bd186c08-5f51-446d-abfc-1d8baff2caa8)
![image](https://github.com/user-attachments/assets/2213c8ec-a83a-4587-888b-eef3811b2efe)


## Installation and Usage Section for the **BREAVA** Project:

---

### 🛠️ Installation & Usage (Local Setup)

#### 1. 📦 Clone the Repository

```bash
git clone https://github.com/Breava-Capstone-Project-Laskar-Ai/breava.git
cd breava
```

#### 2. 🐍 Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

> If you don't have it yet, create a `requirements.txt` file and add the following:

```txt
streamlit
pandas
numpy
matplotlib
scikit-learn
tensorflow
joblib
```

#### 4. 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

#### 5. 🌐 Open in Browser

Go to:

```
http://localhost:8501
```

