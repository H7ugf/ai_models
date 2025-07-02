import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# تحميل البيانات
try:
    data = pd.read_csv('C:/Users/pc/Desktop/homowrk_data_mining/التجميع/cluseter_homurk/cleandata/student_sleep_patterns.csv')
    print("تم تحميل البيانات بنجاح.")
except Exception as e:
    print(f"خطأ في تحميل البيانات: {e}")

# --- 1. فحص القيم الناقصة ---
plt.figure(figsize=(10, 5))
sns.barplot(x=data.isnull().sum().index, y=data.isnull().sum().values)
plt.title('القيم الناقصة قبل المعالجة')
plt.xticks(rotation=45)
plt.show()

# معالجة القيم الناقصة
imputer = SimpleImputer(strategy='median')
numerical_cols = data.select_dtypes(include=['number']).columns
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# التأكد من إزالة القيم الناقصة
plt.figure(figsize=(10, 5))
sns.barplot(x=data.isnull().sum().index, y=data.isnull().sum().values)
plt.title('القيم الناقصة بعد المعالجة')
plt.xticks(rotation=45)
plt.show()

# --- 2. فحص القيم المكررة ---
duplicates = data.duplicated().sum()
if duplicates > 0:
    print(f"يوجد {duplicates} صفوف مكررة. سيتم إزالتها.")
    data = data.drop_duplicates()
else:
    print("لا توجد صفوف مكررة.")

# --- 3. ترميز المتغيرات الفئوية ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=data)
plt.title('توزيع Gender قبل الترميز')
plt.show()

# الترميز
data = pd.get_dummies(data, columns=['Gender'], drop_first=False)
year_mapping = {'1st Year': 1, '2nd Year': 2, '3rd Year': 3, '4th Year': 4}
data['University_Year'] = data['University_Year'].map(year_mapping)

# --- 4. معالجة القيم الشاذة باستخدام IQR ---
numerical_features = [
    'Age', 'Sleep_Duration', 'Study_Hours', 'Screen_Time',
    'Caffeine_Intake', 'Physical_Activity', 'Sleep_Quality'
]

# رسم Boxplot قبل المعالجة
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numerical_features])
plt.title('القيم الشاذة قبل المعالجة')
plt.show()

# معالجة القيم الشاذة
def handle_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    return np.clip(col, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

data[numerical_features] = data[numerical_features].apply(handle_outliers)

# رسم Boxplot بعد المعالجة
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numerical_features])
plt.title('القيم الشاذة بعد المعالجة')
plt.show()

# --- 5. توحيد المقياس باستخدام MinMaxScaler ---
plt.figure(figsize=(10, 5))
sns.histplot(data['Sleep_Duration'], kde=True)
plt.title('توزيع Sleep_Duration قبل التدرج')
plt.show()

# تطبيق التدرج
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.select_dtypes(include=['number']))
cleaned_data = pd.DataFrame(scaled_features, columns=data.select_dtypes(include=['number']).columns)

# إضافة الأعمدة غير العددية
non_numerical = data.select_dtypes(exclude=['number'])
cleaned_data = pd.concat([cleaned_data, non_numerical.reset_index(drop=True)], axis=1)

# رسم توزيع Sleep_Duration بعد التدرج
plt.figure(figsize=(10, 5))
sns.histplot(cleaned_data['Sleep_Duration'], kde=True)
plt.title('توزيع Sleep_Duration بعد التدرج')
plt.show()

# حفظ البيانات المُنظفة
try:
    cleaned_data.to_csv('cleaned_student_data.csv', index=False)
    print("تم حفظ البيانات المُنظفة بنجاح.")
except Exception as e:
    print(f"خطأ في حفظ البيانات: {e}")