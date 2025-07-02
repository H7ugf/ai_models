# 🧑‍🎓 تحليل أنماط نوم الطلاب والتعرف على السعادة من الصور | Student Sleep Patterns & Face Happiness Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

</div>

## 📝 Table of Contents
- [English](#english)
  - [Description](#description)
  - [Features](#features)
  - [Technologies](#technologies)
  - [Installation](#installation)
- [العربية](#العربية)
  - [الوصف](#الوصف)
  - [المميزات](#المميزات)
  - [التقنيات](#التقنيات)
  - [التثبيت](#التثبيت)
- [👨‍💻 Developer](#-developer)
- [📄 License](#-license)

---

## English

### Description
A data science project for analyzing university students' sleep patterns using clustering (K-Means, DBSCAN) and building a deep learning model to recognize happiness from facial images. The project includes data cleaning, clustering, image processing, model training, and a simple web app for face happiness prediction.

### Features
✨ **Data Analysis & Clustering**
- Clean and preprocess student sleep data
- Cluster students based on sleep and study patterns (K-Means, DBSCAN)
- Visualize clusters and statistics

😊 **Face Happiness Recognition**
- Prepare and process face images
- Train a CNN model to classify happy/not happy faces
- Simple Flask web app for real-time prediction

🚀 **Technical Features**
- Data cleaning (missing values, outliers)
- Model training and evaluation
- Interactive web interface

### Technologies
- **Data Science**: Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Deep Learning**: TensorFlow, Keras
- **Web**: Flask
- **Image Processing**: OpenCV, Pillow

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/student-sleep-face-happiness.git
cd student-sleep-face-happiness
```
2. **Install requirements**
```bash
pip install -r requirements.txt
```
3. **Run data cleaning**
```bash
cd cluseter_homurk/cleandata
python clean_data_student_sleep.py
```
4. **Run clustering**
```bash
cd ..
python clusters.py
```
5. **Prepare face image data**
```bash
cd face_recognition
python procces_images.py
```
6. **Train face happiness model**
```bash
python project_python_face.py
```
7. **Run the web app**
```bash
python app.py
```
Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## العربية

### الوصف
مشروع علم بيانات لتحليل أنماط نوم الطلاب الجامعيين باستخدام تقنيات التجميع (K-Means, DBSCAN) وبناء نموذج ذكاء اصطناعي للتعرف على السعادة من تعابير الوجه. يشمل المشروع تنظيف البيانات، التجميع، معالجة الصور، تدريب النموذج، وتطبيق ويب بسيط للتجربة.

### المميزات
✨ **تحليل البيانات والتجميع**
- تنظيف ومعالجة بيانات نوم الطلاب
- تجميع الطلاب حسب أنماط النوم والدراسة (K-Means, DBSCAN)
- رسم وتحليل النتائج

😊 **التعرف على السعادة من الصور**
- تجهيز ومعالجة صور الوجوه
- تدريب نموذج CNN لتصنيف السعادة
- تطبيق ويب Flask للتجربة الفورية

🚀 **ميزات تقنية**
- معالجة القيم الناقصة والشاذة
- تدريب وتقييم النماذج
- واجهة تفاعلية عبر الويب

### التقنيات
- **علم البيانات**: Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **التعلم العميق**: TensorFlow, Keras
- **الويب**: Flask
- **معالجة الصور**: OpenCV, Pillow

### التثبيت
1. **استنساخ المستودع**
```bash
git clone https://github.com/yourusername/student-sleep-face-happiness.git
cd student-sleep-face-happiness
```
2. **تثبيت المتطلبات**
```bash
pip install -r requirements.txt
```
3. **تنظيف البيانات**
```bash
cd cluseter_homurk/cleandata
python clean_data_student_sleep.py
```
4. **تشغيل التجميع**
```bash
cd ..
python clusters.py
```
5. **تجهيز بيانات صور الوجوه**
```bash
cd face_recognition
python procces_images.py
```
6. **تدريب نموذج السعادة**
```bash
python project_python_face.py
```
7. **تشغيل تطبيق الويب**
```bash
python app.py
```
ثم افتح [http://127.0.0.1:5000/](http://127.0.0.1:5000/) في متصفحك.

---

## 👨‍💻 Developer
- **Name**: م محمد علي حزام
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

## 📄 License
This project is licensed for educational and research purposes only. 