import pandas as pd  # استيراد مكتبة Pandas للتعامل مع البيانات.
import numpy as np  # استيراد مكتبة NumPy للعمليات الرياضية على المصفوفات.
import matplotlib.pyplot as plt  # استيراد مكتبة Matplotlib لرسم البيانات.
import seaborn as sns  # استيراد مكتبة Seaborn لتحسين الرسوم البيانية.
import random  # استيراد مكتبة Random لتوليد أرقام عشوائية.
import h5py  # استيراد مكتبة HDF5 للتعامل مع ملفات البيانات.
from keras.models import Sequential, load_model  # استيراد نموذج Keras للتعلم العميق.
#Conv2D :الوظيفة: طبقة الالتفاف (Convolutional Layer) تُستخدم لاستخراج الميزات من الصور.
#MaxPooling2D:الوظيفة: طبقة تجميع (Pooling Layer) تُستخدم لتقليل الأبعاد المكانية للبيانات (الارتفاع والعرض).
#Dense:الوظيفة: طبقة كثيفة (Fully Connected Layer) تُستخدم في الشبكات العصبية لتجميع الميزات المستخرجة.
#Flatten:الوظيفة: طبقة تحويل (Flattening Layer) تستخدم لتحويل البيانات من شكل متعدد الأبعاد إلى شكل أحادي البعد.
# Dropout:طبقة إسقاط (drop) نسبة معينة من الوحدات أثناء التدريب، مما يساعد في تحسين
# BatchNormalization  :تقوم بتطبيع المدخلات لكل طبقة بحيث يكون لها متوسط 0 وانحراف معياري 1، مما يساعد في تقليل تأثير التغيرات في المدخلات.
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization  # استيراد الطبقات المستخدمة في الشبكة العصبية.
from keras.optimizers import Adam  # استيراد خوارزمية Adam للتحديثات.
from keras.preprocessing.image import ImageDataGenerator  # استيراد مولد البيانات لزيادة البيانات.
from sklearn.metrics import confusion_matrix, classification_report  # استيراد مقاييس الأداء مثل مصفوفة الارتباك والتقرير التصنيفي.

# تحميل البيانات
filename_train = 'C:/Users/pc\Desktop/level_four2_IT/face_recognition/train_happy.h5'  # مسار ملف بيانات التدريب.
filename_test = 'C:/Users/pc\Desktop/level_four2_IT/face_recognition/test_happy.h5'    # مسار ملف بيانات الاختبار.

# فتح ملفات HDF5
happy_training = h5py.File(filename_train, 'r')  # فتح ملف التدريب في وضع القراءة.
happy_testing = h5py.File(filename_test, 'r')    # فتح ملف الاختبار في وضع القراءة.

# قراءة البيانات من الملفات
X_train = np.array(happy_training['train_set_x'][:])  # تحويل مجموعة بيانات التدريب إلى مصفوفة NumPy.
y_train = np.array(happy_training['train_set_y'][:])  # تحويل تسميات التدريب إلى مصفوفة NumPy.

X_test = np.array(happy_testing['test_set_x'][:])  # تحويل مجموعة بيانات الاختبار إلى مصفوفة NumPy.
y_test = np.array(happy_testing['test_set_y'][:])  # تحويل تسميات الاختبار إلى مصفوفة NumPy.

# عرض شكل البيانات
print("X_train shape:", X_train.shape)  # طباعة شكل مجموعة بيانات التدريب.
print("y_train shape:", y_train.shape)  # طباعة شكل تسميات التدريب.
print("X_test shape:", X_test.shape)    # طباعة شكل مجموعة بيانات الاختبار.
print("y_test shape:", y_test.shape)    # طباعة شكل تسميات الاختبار.

# عرض صورة عشوائية من مجموعة التدريب
i = random.randint(0, len(X_train) - 1)  # اختيار فهرس عشوائي للصورة.
plt.figure(figsize=(6, 6))  # تحديد حجم الشكل.
plt.imshow(X_train[i])  # عرض الصورة المختارة.
plt.title(f'Label: {y_train[i]}')  # إضافة عنوان يعرض التسمية.
plt.axis('off')  # إزالة المحاور.
plt.show()  # عرض الصورة.

# عرض مجموعة من الصور
W_grid = 5  # عدد الأعمدة في الشبكة.
L_grid = 5  # عدد الصفوف في الشبكة.
n_training = len(X_train)  # عدد الصور في مجموعة التدريب.

fig, axes = plt.subplots(L_grid, W_grid, figsize=(25, 25))  # إنشاء شبكة فرعية لعرض الصور.
axes = axes.ravel()  # تحويل المصفوفة إلى شكل مسطح.

for i in range(W_grid * L_grid):  # تكرار لتوليد صور عشوائية.
    index = np.random.randint(0, n_training)  # اختيار فهرس عشوائي للصورة.
    axes[i].imshow(X_train[index])  # عرض الصورة.
    axes[i].set_title(f'Label: {y_train[index]}', fontsize=25)  # إضافة عنوان للصورة.
    axes[i].axis('off')  # إزالة المحاور.

plt.suptitle('Sample Images from Training Set', fontsize=30)  # عنوان عام للشبكة.
plt.show()  # عرض الصور.

# تطبيع البيانات
X_train = X_train / 255.0  # تطبيع قيم بيانات التدريب إلى نطاق [0, 1].
X_test = X_test / 255.0    # تطبيع قيم بيانات الاختبار إلى نطاق [0, 1].

# إعداد زيادة البيانات
datagen = ImageDataGenerator(  # إعداد مولد البيانات لزيادة البيانات.
    rotation_range=20,  # دوران الصورة بزاوية حتى 20 درجة.
    width_shift_range=0.2,  # نقل الصورة عرضياً حتى 20%.
    height_shift_range=0.2,  # نقل الصورة عمودياً حتى 20%.
    shear_range=0.2,  # قص الصورة بزاوية حتى 20%.
    zoom_range=0.2,  # تكبير أو تصغير الصورة حتى 20%.
    horizontal_flip=True,  # عكس الصورة أفقياً.
    fill_mode='nearest'  # طريقة ملء المساحات الفارغة.
)

# بناء نموذج CNN
cnn_model = Sequential()  # إنشاء نموذج تسلسلي.
cnn_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # طبقة الالتفاف.
cnn_model.add(BatchNormalization())  # تطبيع الدفعة لتحسين التدريب.
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))  # طبقة تجميع.
cnn_model.add(Dropout(0.25))  # طبقة الإسقاط لتقليل الإفراط في التكيف.
# حيث تُرجع القيمة 0 إذا كانت 
# 𝑥
# x أقل من 0، وأما إذا كانت 
# x أكبر أو تساوي 0، فإنها تُرجع القيمة نفسها 

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))  # طبقة الالتفاف الثانية.
cnn_model.add(BatchNormalization())  # تطبيع الدفعة.
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))  # طبقة تجميع ثانية.
cnn_model.add(Dropout(0.25))  # طبقة إسقاط ثانية.
#تحسين القدرة على التعميم
#زيادة تعقيد النموذج

cnn_model.add(Flatten())  # تحويل البيانات إلى شكل مسطح.
cnn_model.add(Dense(128, activation='relu'))  # طبقة كثيفة مع 128 وحدة.
cnn_model.add(Dense(1, activation='sigmoid'))  # طبقة الإخراج (لثنائية التصنيف).
#sigmoid:النطاق: قيمة دالة Sigmoid تتراوح بين 0 و 1. هذا يجعلها مناسبة لمهام التصنيف الثنائي، حيث يمكن اعتبار الخرج كاحتمال.
# تجميع النموذج
cnn_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # تجميع النموذج مع تحديد الخسارة والمقياس المستخدم.

# تدريب النموذج مع زيادة البيانات
epochs = 10  # عدد epochs للتدريب.
history = cnn_model.fit(datagen.flow(X_train, y_train, batch_size=30), epochs=epochs, verbose=1)  # تدريب النموذج مع بيانات تم زيادتها.
#  batch_size=30:يعني أنه سيتم استخدام 30 عينة في كل خطوة تدريب.
# توقع الفئات
# verbose=1:
# يتم عرض معلومات تفصيلية في كل فترة (epoch)، بما في ذلك:
# عدد الفترة الحالية.
# قيمة الخسارة (loss).
# قياسات الأداء مثل الدقة (accuracy).
predicted_classes = (cnn_model.predict(X_test) > 0.5).astype("int32")  # توقع الفئات بناءً على عتبة 0.5.

# مصفوفة الارتباك
cm = confusion_matrix(y_test, predicted_classes)  # حساب مصفوفة الارتباك.
plt.figure(figsize=(10, 7))  # تحديد حجم الشكل.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # رسم مصفوفة الارتباك.
plt.title('Confusion Matrix', fontsize=20)  # عنوان للمصفوفة.
plt.xlabel('Predicted Label', fontsize=15)  # عنوان المحور السيني.
plt.ylabel('True Label', fontsize=15)  # عنوان المحور الصادي.
plt.show()  # عرض المصفوفة.

# تقرير التصنيف
print(classification_report(y_test, predicted_classes))  # طباعة تقرير التصنيف الذي يشمل الدقة، الاستدعاء، والنقاط الأخرى.

# تحليل أداء النموذج
plt.figure(figsize=(12, 5))  # تحديد حجم الشكل.
plt.subplot(1, 2, 1)  # إنشاء الشكل الفرعي الأول.
plt.plot(history.history['accuracy'], label='Train Accuracy')  # رسم دقة التدريب على مر العصور.
plt.title('Model Accuracy')  # عنوان الشكل.
plt.xlabel('Epoch')  # عنوان المحور السيني.
plt.ylabel('Accuracy')  # عنوان المحور الصادي.
plt.legend()  # إضافة وسيلة إيضاح.

plt.subplot(1, 2, 2)  # إنشاء الشكل الفرعي الثاني.
plt.plot(history.history['loss'], label='Train Loss')  # رسم خسارة التدريب على مر العصور.
plt.title('Model Loss')  # عنوان الشكل.
plt.xlabel('Epoch')  # عنوان المحور السيني.
plt.ylabel('Loss')  # عنوان المحور الصادي.
plt.legend()  # إضافة وسيلة إيضاح.

plt.tight_layout()  # ضبط التخطيط.
plt.show()  # عرض الرسوم البيانية.

# عرض بعض نتائج التوقعات
L = 5  # عدد الصفوف في الشبكة.
W = 5  # عدد الأعمدة في الشبكة.
fig, axes = plt.subplots(L, W, figsize=(12, 12))  # إنشاء شبكة فرعية للعرض.
axes = axes.ravel()  # تحويل المصفوفة إلى شكل مسطح.

for i in range(L * W):  # تكرار لتوليد صور توقعات.
    axes[i].imshow(X_test[i])  # عرض الصورة.
    axes[i].set_title(f'Pred: {predicted_classes[i][0]}, True: {y_test[i]}', fontsize=12)  # إضافة عنوان يعرض التوقع والواقع.
    axes[i].axis('off')  # إزالة المحاور.

plt.subplots_adjust(wspace=0.5)  # تعديل المسافات بين الصور.
plt.suptitle('Sample Predictions', fontsize=20)  # عنوان عام.
plt.show()  # عرض الصور.

# عرض الصور التي تم تصنيفها بشكل خاطئ
incorrect_indices = np.where(predicted_classes.flatten() != y_test)[0]  # الحصول على فهارس الصور المصنفة بشكل خاطئ.
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # إنشاء شبكة فرعية لعرض الأخطاء.
for ax, index in zip(axes.flatten(), incorrect_indices[:10]):  # عرض أول 10 أخطاء.
    ax.imshow(X_test[index])  # عرض الصورة.
    ax.set_title(f'Pred: {predicted_classes[index][0]}, True: {y_test[index]}', fontsize=12)  # إضافة عنوان للتوقع والواقع.
    ax.axis('off')  # إزالة المحاور.
plt.suptitle('Incorrect Predictions', fontsize=20)  # عنوان عام.
plt.show()  # عرض الصور.

# حفظ النموذج
#cnn_model.save('C:/Users/pc/Desktop/face_recognition/my_save_model.h5')  # حفظ النموذج في ملف HDF5 (تم التعليق عليه).