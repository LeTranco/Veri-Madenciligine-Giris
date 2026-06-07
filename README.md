# Piston Halkaları Taşlama İşlemi Makine Öğrenmesi Uygulaması

BLM0463 Veri Madenciliğine Giriş dersinin dönem ödevi için hazırladığım bu projede amaç, sadece makine ayarlarına ve sensör okumalarına bakarak o an üretilen parçanın
kalitesinin ne olacağını, parçayı daha test etmeden önce otomatik olarak tahmin etmektir.

## Model Değerlendirmesi

Projede eğitilen Gaussian Naïve Bayes modelinin performansı, hem istatistiksel metrikler hem de çeşitli görselleştirme araçları kullanılarak kapsamlı bir şekilde değerlendirilmiştir.

### 1. Temel Performans Metrikleri

Test seti üzerinde yapılan tahminler sonucunda elde edilen temel sınıflandırma başarı ölçütleri şu şekildedir:

* **Accuracy:** `0.529`
* **Sensitivity:** `0.577`
* **Specificity:** `0.479`

### 2. Genel Tutarlılık
Modelin kararlılığını test etmek ve ezberleme durumunun önüne geçmek amacıyla 5 katmanlı çapraz doğrulama uygulanmıştır:
* **Katman Başarıları:** `0.4958, 0.5083, 0.5166, 0.4916, 0.4916`
* **Ortalama Güvenilir Doğruluk:** `0.5008`

### 3. Confusion Matrix
Sınıflandırıcının hangi sınıfları ne kadar doğru veya hatalı tahmin ettiğini gösteren kafa karışıklığı matrisi aşağıda yer almaktadır.

<img width="800" height="600" alt="2_confusion_matrix" src="https://github.com/user-attachments/assets/c5241d23-488b-444e-8278-8d6d09ba145f" />

### 4. ROC Eğrisi
Modelin farklı sınıflandırma eşik değerlerindeki başarımını gösteren ROC Eğrisi çizdirilmiştir.

<img width="640" height="480" alt="3_roc_egrisi" src="https://github.com/user-attachments/assets/551a9cd2-4175-4c45-b815-eff704197acb" />

### 5. Özellik Önemi
Modelin tahmin kararlarını verirken hangi bağımsız değişkenlere daha çok güvendiğini saptamak amacıyla permütasyon önem dereceleri hesaplanmıştır.

<img width="1000" height="600" alt="4_ozellik_onem_grafigi" src="https://github.com/user-attachments/assets/9c7e9d7d-12e8-4c78-9911-7cbb67042662" />

## Kaynakça

M. de Castro Cesário, M. V. de Souza, M. C. Pereira, and A. P. de Paiva, "Industrial manufacturing dataset for cylindrical plunge grinding of martensitic gray cast iron piston rings," Data in Brief, vol. 66, art. no. 112706, Jun. 2026. 
