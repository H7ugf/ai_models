import h5py  # مكتبة للتعامل مع ملفات HDF5
import numpy as np  # مكتبة للعمليات الرياضية على المصفوفات
from PIL import Image  # مكتبة لمعالجة الصور
import os  # مكتبة للتعامل مع نظام الملفات
import random  # مكتبة لتوليد أرقام عشوائية

# تعيين أسماء ملفات HDF5
train_filename = 'train_happy.h5'  # اسم ملف بيانات التدريب
test_filename = 'test_happy.h5'     # اسم ملف بيانات الاختبار

# تحديد مسار المجلد الذي يحتوي على الصور
image_folder = 'C:/Users/pc/Desktop/homowrk_data_mining/التجميع/cluseter_homurk/face_recognition/New_folder'  # تأكد من تعديل هذا المسار حسب موقع مجلد الصور
images = []  # قائمة لتخزين الصور
labels = []  # قائمة لتخزين التسميات

# قراءة الصور من المجلد
for file in os.listdir(image_folder):  # استعراض جميع الملفات في المجلد
    if file.endswith('.jpg') or file.endswith('.png'):  # تحقق مما إذا كان الملف صورة
        # قراءة الصورة
        img_path = os.path.join(image_folder, file)  # الحصول على المسار الكامل للصورة
        img = Image.open(img_path)  # فتح الصورة

        # تغيير حجم الصورة إلى 64x64 بكسل
        img = img.resize((64, 64), Image.LANCZOS)  # تغيير حجم الصورة مع تحسين الجودة
        
        img_array = np.array(img) / 255.0  # تحويل الصورة إلى مصفوفة وتطبيع القيم إلى [0, 1]
        
        # التحقق من الأبعاد
        if img_array.shape == (64, 64, 3):  # تأكد من أن الصورة ملونة (RGB)
            images.append(img_array)  # إضافة الصورة إلى القائمة
            labels.append(1 if 'happy' in file else 0)  # إضافة التسمية

# تحويل القوائم إلى مصفوفات NumPy
images_array = np.array(images)  # تحويل قائمة الصور إلى مصفوفة NumPy
labels_array = np.array(labels)  # تحويل قائمة التسميات إلى مصفوفة NumPy

# تقسيم البيانات إلى مجموعة تدريب واختبار (80% تدريب، 20% اختبار)
train_size = int(0.8 * len(images_array))  # تحديد حجم مجموعة التدريب (80%)
indices = list(range(len(images_array)))  # إنشاء قائمة بأرقام الفهارس
random.shuffle(indices)  # خلط الفهارس عشوائياً

# تقسيم البيانات
train_indices = indices[:train_size]  # الحصول على فهارس مجموعة التدريب
test_indices = indices[train_size:]  # الحصول على فهارس مجموعة الاختبار

X_train = images_array[train_indices]  # مصفوفة بيانات التدريب
y_train = labels_array[train_indices]  # مصفوفة تسميات التدريب
X_test = images_array[test_indices]  # مصفوفة بيانات الاختبار
y_test = labels_array[test_indices]  # مصفوفة تسميات الاختبار

# طباعة عدد الصور والتسميات
print(f"عدد صور التدريب: {X_train.shape[0]}")  # طباعة عدد صور التدريب
print(f"عدد تسميات التدريب: {y_train.shape[0]}")  # طباعة عدد تسميات التدريب
print(f"عدد صور الاختبار: {X_test.shape[0]}")  # طباعة عدد صور الاختبار
print(f"عدد تسميات الاختبار: {y_test.shape[0]}")  # طباعة عدد تسميات الاختبار

# حفظ بيانات التدريب في ملف HDF5
with h5py.File(train_filename, 'w') as h5file:  # فتح ملف HDF5 للتدريب للكتابة
    h5file.create_dataset('train_set_x', data=X_train)  # إنشاء مجموعة بيانات لصور التدريب
    h5file.create_dataset('train_set_y', data=y_train)  # إنشاء مجموعة بيانات لتسميات التدريب

# حفظ بيانات الاختبار في ملف HDF5
with h5py.File(test_filename, 'w') as h5file:  # فتح ملف HDF5 للاختبار للكتابة
    h5file.create_dataset('test_set_x', data=X_test)  # إنشاء مجموعة بيانات لصور الاختبار
    h5file.create_dataset('test_set_y', data=y_test)  # إنشاء مجموعة بيانات لتسميات الاختبار

# التحقق من حفظ البيانات
with h5py.File(train_filename, 'r') as h5file:  # فتح ملف HDF5 للتدريب للقراءة
    print("المفاتيح في ملف HDF5 للتدريب:")  # طباعة رسالة
    print(list(h5file.keys()))  # طباعة المفاتيح الموجودة في الملف
    print("شكل مجموعة بيانات التدريب:", h5file['train_set_x'].shape)  # طباعة شكل مجموعة بيانات التدريب
    print("شكل مجموعة بيانات تسميات التدريب:", h5file['train_set_y'].shape)  # طباعة شكل مجموعة بيانات تسميات التدريب

with h5py.File(test_filename, 'r') as h5file:  # فتح ملف HDF5 للاختبار للقراءة
    print("المفاتيح في ملف HDF5 للاختبار:")  # طباعة رسالة
    print(list(h5file.keys()))  # طباعة المفاتيح الموجودة في الملف
    print("شكل مجموعة بيانات الاختبار:", h5file['test_set_x'].shape)  # طباعة شكل مجموعة بيانات الاختبار
    print("شكل مجموعة بيانات تسميات الاختبار:", h5file['test_set_y'].shape)  # طباعة شكل مجموعة بيانات تسميات الاختبار

# التحقق من وجود الملفات
for fname in [train_filename, test_filename]:  # استعراض أسماء الملفات
    if os.path.exists(fname):  # تحقق مما إذا كان الملف موجوداً
        print(f"الملف تم حفظه في: {os.path.abspath(fname)}")  # طباعة المسار الكامل للملف
    else:
        print("الملف غير موجود.")  # طباعة رسالة إذا كان الملف غير موجود
