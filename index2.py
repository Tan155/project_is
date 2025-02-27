import tensorflow as tf

# โหลดโมเดลเดิม
model = tf.keras.models.load_model("glasses_model.h5")

# แปลงโมเดลเป็น TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# บันทึกไฟล์ที่ถูกบีบอัด
with open("glasses_model.tflite", "wb") as f:
    f.write(tflite_model)
