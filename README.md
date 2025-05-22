Etiket:#ttg5hackathon2025

# Gelişmiş Göz Pedleri Kalite Kontrol Sistemi

![Eye Pad Quality Control Demo](https://via.placeholder.com/600x400?text=Proje+Görseli+Eklenecek) 
*(Buraya projenizden bir örnek görsel veya GIF ekleyebilirsiniz, örn. modelin algılama yaptığı bir görsel)*

## 🚀 Proje Genel Bakışı

Bu proje, yüksek hassasiyetli Ultralytics YOLOv8 nesne algılama modelini kullanarak **göz pedlerinin üretim hattındaki kalite kusurlarını otomatik olarak tespit etmek** amacıyla geliştirilmiştir. Amacımız, manuel denetimdeki insan hatalarını en aza indirerek ve kusur tespit süreçlerini otomatikleştirerek üretim verimliliğini artırmaktır.

Hackathon sürecinde tasarlanan ve geliştirilen bu sistem, yapay zeka ve bilgisayar görüşü alanındaki bilgi birikimimizin bir ürünüdür.

## ✨ Temel Özellikler

* **YOLOv8 Destekli Hassas Algılama:** Sektör standardı YOLOv8 mimarisi ile güçlü ve gerçek zamanlı kusur tespiti.
* **Özelleştirilmiş Kusur Sınıfları:** Göz pedlerine özgü aşağıdaki kritik kusur tiplerini başarıyla algılar:
    * `saglam_ped` (Sağlam ped)
    * `renk_degisimi` (Renk değişimi)
    * `leke` (Leke)
    * `yapi_bozulmasi` (Yapı bozulması)
    * `kesim_hatasi` (Kesim hatası)
* **Çoklu Platform Uyumluluğu:** Eğitilmiş model, farklı dağıtım (deployment) ortamları için optimize edilmiş formatlara dönüştürülebilir:
    * **ONNX (.onnx):** Çeşitli programlama dilleri ve runtime'lar (C++, Java, .NET, Rust vb.) ile geniş uyumluluk.
    * **TensorFlow Lite (TFLite - .tflite):** Android cihazlar ve gömülü sistemler için optimize edilmiş, hafif ve hızlı çıkarım (FP32, FP16 ve INT8 nicemleme seçenekleriyle).
* **Özel Veri Seti:** Gerçek dünya üretim koşullarını yansıtan, özenle etiketlenmiş özel bir göz pedleri veri seti üzerinde eğitim.

## 📁 Depo Yapısı

Proje dosyaları aşağıdaki temel yapıya sahiptir:

.
├── eye_pad_quality_control/
│   ├── yolov8n_custom_dataset/
│   │   ├── images/  # Eğitim ve doğrulama görselleri
│   │   ├── labels/  # YOLO formatındaki etiketler
│   │   └── weights/ # Eğitilmiş model ağırlıkları (best.pt, best.onnx, best.tflite)
│   └── data.yaml    # Veri seti yapılandırma dosyası
├── train.py         # Model eğitim betiği (veya Jupyter Notebook içinde)
├── inference.py     # Model ile tahmin yapma betiği (Örnek)
├── convert_models.py # Model dönüşüm betiği (ONNX, TFLite)
├── README.md        # Bu dosya
└── requirements.txt # Proje bağımlılıkları


## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırabilmek için aşağıdaki adımları takip edin:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/ErennOzturkk/Cakal207-ttg5hackathon2025.git](https://github.com/ErennOzturkk/Cakal207-ttg5hackathon2025.git)
    cd Cakal207-ttg5hackathon2025
    ```

2.  **Sanal Ortam Oluşturun (Şiddetle Tavsiye Edilir):**
    Bağımlılık çakışmalarını önlemek için yeni bir Python sanal ortamı oluşturun ve etkinleştirin.

    ```bash
    python -m venv yolov8_env
    .\yolov8_env\Scripts\activate # Windows için
    # source yolov8_env/bin/activate # Linux/macOS için
    ```

3.  **Bağımlılıkları Yükleyin:**
    `requirements.txt` dosyanız varsa, oradan yükleyebilirsiniz. Yoksa, aşağıdaki komutları kullanarak gerekli tüm kütüphaneleri belirtilen uyumlu sürümlerle yükleyin:

    ```bash
    pip install --upgrade pip
    pip install ultralytics==8.3.141
    pip install protobuf==3.20.1
    pip install tensorflow==2.10.0 # Python 3.9 ile uyumlu CPU versiyonu
    pip install onnx==1.17.0 onnxruntime onnxslim
    pip install tf_keras
    pip install sng4onnx>=1.0.1
    pip install onnx_graphsurgeon>=0.3.26
    pip install ai-edge-litert>=1.2.0
    pip install onnx2tf>=1.26.3
    ```
    *Not: TensorFlow sürümü Python sürümünüze ve CUDA/GPU durumunuza göre değişebilir. Eğer `tensorflow==2.10.0` ile sorun yaşarsanız, sadece `pip install tensorflow` deneyip sonrasında `protobuf`'u tekrar `3.20.1`'e düşürmeyi deneyebilirsiniz.*

## 🚀 Kullanım

### Veri Seti Hazırlığı

Veri setinizi `eye_pad_quality_control/yolov8n_custom_dataset` dizini altında aşağıdaki yapıya uygun şekilde düzenleyin:

eye_pad_quality_control/yolov8n_custom_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
├── train/
└── val/
└── test/


`data.yaml` dosyanız (örnek: `eye_pad_quality_control/data.yaml`), veri setinizin yollarını ve sınıf isimlerini tanımlamalıdır:

```yaml
# data.yaml içeriği
path: ../eye_pad_quality_control/yolov8n_custom_dataset/  # Veri setinizin ana yolu
train: images/train # Eğitim görselleri
val: images/val     # Doğrulama görselleri
test: images/test   # Test görselleri (opsiyonel)

names:
  0: saglam_ped
  1: renk_degisimi
  2: leke
  3: yapi_bozulmasi
  4: kesim_hatasi
Modeli Eğitme
Eğitim sürecini başlatmak için train.py dosyasını (veya Jupyter Notebook içinde ilgili hücreleri) kullanın:

Python

# train.py veya Jupyter Notebook hücresi
from ultralytics import YOLO

# Modeli başlatın (önceden eğitilmiş 'n' nano modeli ile başlayabiliriz)
model = YOLO('yolov8n.pt') 

# Modeli özel veri setinizde eğitin
results = model.train(data='eye_pad_quality_control/data.yaml', epochs=50, imgsz=640, device='cpu')
# GPU kullanmak için 'device=0' (veya uygun GPU ID) ayarlayabilirsiniz.
Modeli Değerlendirme
Eğitilen modelin performansını doğrulama seti üzerinde değerlendirin:

Python

# inference.py veya Jupyter Notebook hücresi
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('eye_pad_quality_control/yolov8n_custom_dataset/weights/best.pt') 

# Modelin performansını değerlendir
metrics = model.val()

print(f"Mean Average Precision (mAP@0.50-0.95): {metrics.results_dict['metrics/mAP50-95(B)']}")
print(f"Mean Average Precision (mAP@0.50): {metrics.results_dict['metrics/mAP50(B)']}")
Çıkarım (Prediction) Yapma
Yeni görüntüler üzerinde tahminler yapmak için:

Python

# inference.py veya Jupyter Notebook hücresi
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('eye_pad_quality_control/yolov8n_custom_dataset/weights/best.pt') 

# Tek bir görüntüde tahmin yap
results = model('path/to/your/image.jpg', conf=0.25, iou=0.7) # Güven eşiği 0.25, IOU eşiği 0.7

# Sonuçları görselleştir veya kaydet
for r in results:
    im_bgr = r.plot()  # Çıkarım kutularını ve etiketleri görüntüye çiz
    # cv2.imshow('Prediction', im_bgr) # Eğer OpenCV kuruluysa ve görselleştirmek istersen
    # cv2.imwrite('predicted_image.jpg', im_bgr) # Kaydetmek için
Model Dönüşümleri
Eğitilmiş modelinizi farklı dağıtım platformları için optimize edilmiş formatlara dönüştürün.

1. ONNX Formatına Dönüştürme
Python

# convert_models.py veya Jupyter Notebook hücresi
from ultralytics import YOLO

model_path = 'eye_pad_quality_control/yolov8n_custom_dataset/weights/best.pt'
model = YOLO(model_path)

print("Model ONNX formatına dönüştürülüyor...")
results_onnx = model.export(format='onnx', imgsz=640, device='cpu', simplify=True)
print(f"ONNX modeli başarıyla kaydedildi: {results_onnx}")
2. TensorFlow Lite (TFLite) Formatına Dönüştürme
Bu süreç, PyTorch modelini önce ONNX'e, oradan TensorFlow SavedModel'a ve son olarak TFLite'a dönüştürme adımlarını içerir.

Python

# convert_models.py veya Jupyter Notebook hücresi
import onnx2tf
import tensorflow as tf
import os
from ultralytics import YOLO
import numpy as np

model_path = 'eye_pad_quality_control/yolov8n_custom_dataset/weights/best.pt'
model = YOLO(model_path)

# 1. Adım: PyTorch (.pt) modelini ONNX'e dönüştür (eğer henüz yapmadıysanız)
print("ONNX model oluşturuluyor...")
# Bu kısım zaten başarıyla yapılmış olmalı, eğer yoksa tekrar çalıştırın
model.export(format='onnx', imgsz=640, device='cpu', simplify=True)
onnx_model_path = 'eye_pad_quality_control/yolov8n_custom_dataset/weights/best.onnx'
print(f"ONNX modeli kaydedildi: {onnx_model_path}")

# 2. Adım: ONNX modelini TensorFlow SavedModel'a dönüştür
output_saved_model_dir = 'eye_pad_quality_control/yolov8n_custom_dataset/weights/saved_model_tf'
print(f"\nONNX modelini TensorFlow SavedModel'a dönüştürülüyor: {output_saved_model_dir}")
onnx2tf.convert(
    input_onnx_file_path=onnx_model_path,
    output_folder_path=output_saved_model_dir,
    non_verbose=True
)
print("TensorFlow SavedModel başarıyla oluşturuldu.")

# 3. Adım: TensorFlow SavedModel'ı TFLite'a dönüştür
# Çıkış TFLite dosya yolları
output_tflite_path_fp32 = os.path.join(output_saved_model_dir, 'best_fp32.tflite')
output_tflite_path_fp16 = os.path.join(output_saved_model_dir, 'best_fp16.tflite')
output_tflite_path_int8 = os.path.join(output_saved_model_dir, 'best_int8.tflite')

# FP32 TFLite Dönüşümü
print("\nTensorFlow SavedModel'dan FP32 TFLite'a dönüştürülüyor...")
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(output_saved_model_dir)
tflite_model_fp32 = converter_fp32.convert()
with open(output_tflite_path_fp32, 'wb') as f:
    f.write(tflite_model_fp32)
print(f"FP32 TFLite modeli başarıyla kaydedildi: {output_tflite_path_fp32}")

# FP16 TFLite Dönüşümü (Opsiyonel)
print("\nTensorFlow SavedModel'dan FP16 TFLite'a dönüştürülüyor...")
converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(output_saved_model_dir)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_model_fp16 = converter_fp16.convert()
with open(output_tflite_path_fp16, 'wb') as f:
    f.write(tflite_model_fp16)
print(f"FP16 TFLite modeli başarıyla kaydedildi: {output_tflite_path_fp16}")
 INT8 TFLite Dönüşümü (Opsiyonel - Temsili Veri Kümesi Gerektirir)
print("\nTensorFlow SavedModel'dan INT8 TFLite'a dönüştürülüyor...")
converter_int8 = tf.lite.TFLiteConverter.from_saved_model(output_saved_model_dir)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8 # Giriş tipi uint8 olarak ayarlanabilir
converter_int8.inference_output_type = tf.uint8 # Çıkış tipi uint8 olarak ayarlanabilir

import numpy as np
def representative_dataset_gen():
    # Gerçek veri setinizden (örn. eğitim veya doğrulama setinden)
    # 20-100 adet normalize edilmiş görüntü örneği sağlamalısınız.
    # Her görüntü (1, 640, 640, 3) boyutunda np.float32 tipinde olmalı.
    # Örneğin:
    # for img_path in your_train_image_paths[:20]:
    #     img = cv2.imread(img_path)
    #     img = cv2.resize(img, (640, 640))
    #     img = img / 255.0 # Normalize to [0, 1]
    #     yield [np.expand_dims(img, axis=0).astype(np.float32)]
    
    # Şimdilik rastgele veri ile örnek veriyoruz:
    for _ in range(20): 
        data = np.random.rand(1, 640, 640, 3).astype(np.float32)
        yield [data]

converter_int8.representative_dataset = representative_dataset_gen
tflite_model_int8 = converter_int8.convert()
with open(output_tflite_path_int8, 'wb') as f:
    f.write(tflite_model_int8)
print(f"INT8 TFLite modeli başarıyla kaydedildi: {output_tflite_path_int8}")

🤝 Katkıda Bulunma
Projeye katkıda bulunmak isterseniz aşağıdaki adımları izleyebilirsiniz:

Bu depoyu (repository) çatallayın (fork).
Yeni bir özellik dalı (feature branch) oluşturun: git checkout -b ozellik/yeni-ozellik
Değişikliklerinizi yapın ve commit edin: git commit -m 'Yeni özellik: Açıklama'
Dalı push edin: git push origin ozellik/yeni-ozellik
Bir Pull Request (Çekme İsteği) oluşturun.
