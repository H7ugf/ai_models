import os  # مكتبة للتعامل مع نظام الملفات
import cv2  # مكتبة OpenCV لمعالجة الصور والفيديو
import numpy as np  # مكتبة للعمليات الرياضية على المصفوفات
#render_template:بإرجاعه كاستجابة للعميل.
#request:مكنك استخدامه للوصول إلى المعلومات المرسلة من العميل، مثل البيانات النصية أو الملفات.
#jsonify:تستخدم هذه الدالة لتحويل البيانات (مثل القوائم أو القواميس) إلى تنسيق JSON وإرجاعها كاستجابة.

from flask import Flask, render_template, request, jsonify  # استيراد مكتبة Flask لإنشاء تطبيق ويب
from keras.models import load_model  # استيراد دالة لتحميل النموذج المدرب من Keras


# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل النموذج المدرب
model = load_model('C:/Users/pc/Desktop/homowrk_data_mining/التجميع/cluseter_homurk/face_recognition/my_save_model.h5')  # استبدل بمسار النموذج الخاص بك

# دالة لتحليل الصورة
def analyze_image(image_data):
    # تحويل بيانات الصورة من البايتات إلى مصفوفة NumPy
    file_bytes = np.frombuffer(image_data, np.uint8)
    # فك تشفير الصورة باستخدام OpenCV
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # تغيير حجم الصورة إلى 64x64 بكسل وتطبيع القيم إلى [0, 1]
    resized_image = cv2.resize(image, (64, 64)) / 255.0
    # إعادة تشكيل الصورة لتتناسب مع مدخلات النموذج
# 1: يمثل عدد الصور. هنا، نحن نعيد تشكيل الصورة كصورة واحدة. هذا مهم لأن نماذج التعلم الآلي عادةً ما تتوقع أن تأتي البيانات في شكل دفعات (batch) من الصور.
# 64, 64, 3: تمثل الأبعاد الثلاثة للصورة (الارتفاع، العرض، وعدد القنوات).
    reshaped_image = np.reshape(resized_image, (1, 64, 64, 3))
    # إجراء التنبؤ باستخدام النموذج المدرب
    prediction = model.predict(reshaped_image)

    # طباعة التنبؤات لمساعدتك في التصحيح
    print("التنبؤات:", prediction)

    # تحديد ما إذا كانت الصورة تعبر عن السعادة أو لا
    return "Happy" if prediction[0][0] > 0.5 else "Not Happy"

# تعريف المسار الرئيسي للتطبيق
@app.route('/')
def index():
    # عرض صفحة الويب الرئيسية
    return render_template('index.html')

# تعريف مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    # قراءة بيانات الصورة المرفقة في الطلب
    image_data = request.files['image'].read()
    # تحليل الصورة وإجراء التنبؤ
    result = analyze_image(image_data)
    # إرجاع النتيجة كاستجابة JSON
    return jsonify({'result': result})

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)  # تشغيل التطبيق في وضع التصحيح
    
    
    
