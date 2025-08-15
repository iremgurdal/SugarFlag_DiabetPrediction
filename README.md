### 📁 Proje Sunumu
Projenin diğer koduna (https://drive.google.com/file/d/1cH6PGW-7dS4X8kN9NJxmZ3uDYPqv_7uI/view?usp=sharing) ulaşabilirsiniz.

SugarFlag: Bireysel Göstergeler ile Diyabet Tahmin Modeli
SugarFlag, bireylerin sağlık göstergelerini kullanarak diyabet riskini tahmin eden gelişmiş bir makine öğrenimi projesidir. Özellikle dengesiz veri setleriyle başa çıkmak için tasarlanmış yenilikçi yaklaşımlar (meta-öğrenme, sınıf ağırlıkları) ile yüksek performans hedeflenmiştir.

👥 Proje Ekibi

Seher Şahan CEYLAN

İrem GÜRDAL

Esra ÇİL

📌 İçindekiler
Proje Hakkında

Veri Seti

Model Yaklaşımı

Dosya Yapısı

Kurulum ve Çalıştırma

Kullanım

Sonuçlar

🚀 Proje Hakkında
Bu projenin temel amacı, diyabet hastalığının erken teşhisi için bireysel sağlık göstergelerine dayanan güvenilir bir makine öğrenimi modeli geliştirmektir. Proje, BRFSS 2015 sağlık göstergeleri veri seti üzerinde çalışır ve aşağıdaki adımları içerir:

Kapsamlı veri ön işleme ve özellik mühendisliği (feature engineering)

Dengesiz sınıflar sorununa yönelik özel çözümler

Birden fazla modeli birleştiren (ensemble/stacking) bir meta modelin oluşturulması

Son modelin, kullanıcı dostu bir Streamlit web uygulaması ile sunulması.

📊 Veri Seti
Veri Kaynağı: BRFSS 2015 Sağlık Göstergeleri

Kayıt Sayısı: 253.680

Özellik Sayısı: 22

Hedef Değişken: Diabetes_012 (0: Diyabetsiz, 1: Pre-diyabet, 2: Diyabet)
Proje, ikili sınıflandırma (diyabet var/yok) problemine odaklandığı için Diabetes_binary adında yeni bir hedef değişken türetilmiştir.

🤖 Model Yaklaşımı
Proje, model performansını maksimize etmek için çok aşamalı bir yaklaşıma sahiptir.

1. Özellik Mühendisliği (Feature Engineering)
Ham veriden daha tahminci özellikler türetilmiştir:

BMI_cat_code, BMI_Age_interaction

HighRisk_Obese_Old gibi yüksek risk gruplarını işaret eden etkileşimli özellikler.

2. Meta Öğrenme (Meta-Learning) Yaklaşımı
Geleneksel tekil model yerine, daha güçlü ve genelleştirilebilir bir meta model mimarisi kullanılmıştır:

Temel Modeller: RandomForestClassifier ve GaussianNB modelleri eğitilir.

Özellik Genişletme: Bu temel modellerin tahmin olasılıkları (predict_proba), yeni özellikler olarak ana veri setine eklenir.

Nihai Model: Genişletilmiş bu veri seti üzerinde, nihai kararı vermek için RandomForestClassifier algoritması ile bir meta model eğitilir. Bu yapı, her bir temel modelin güçlü yönlerinden faydalanır.


3. Dengesiz Veri Çözümü
Azınlık sınıflarının (diyabet) doğru tahmin edilmesi için model eğitiminde sınıf ağırlıkları (class_weight) kullanılmıştır. Bu yöntemle model, yanlış negatifleri (diyabetli birini diyabetsiz olarak tahmin etme) daha fazla cezalandırarak, diyabet teşhisindeki hassasiyeti artırmayı hedefler.

📁 Dosya Yapısı
.
├── App.py                                          # Streamlit web uygulamasının ana dosyası
├── diabetes_012_health_indicators_BRFSS2015.csv    # Ham veri seti
├── meta_model_rf.pkl                               # Eğitilmiş, tüm pipeline'ı içeren model dosyası
├── pipeline_train.py                               # Ön işleme ve model eğitim pipeline'ını oluşturan betik
├── Nihaimodel.py                                   # Farklı model denemeleri ve analizlerin yapıldığı betik
└── SugarFlag-Bireysel-Gostergeler-ile-Diyabet-Tahmini (2).pptx # Proje sunumu

⚙️ Kurulum ve Çalıştırma
Projenin yerel makinenizde çalışması için aşağıdaki adımları takip edebilirsiniz.

Adım 1: Depoyu Klonlama

git clone https://github.com/github_kullanici_adiniz/repo_adiniz.git
cd repo_adiniz
github_kullanici_adiniz ve repo_adiniz kısımlarını kendi bilgilerinizle değiştirmeyi unutmayın.

Adım 2: Gerekli Kütüphaneleri Yükleme
Projenin bağımlılıklarını kurmak için aşağıdaki komutu çalıştırın:

pip install pandas scikit-learn streamlit joblib numpy

Adım 3: Modeli Eğitme
Eğer meta_model_rf.pkl dosyası mevcut değilse, Streamlit uygulamasını çalıştırmadan önce aşağıdaki komutla modeli eğitip kaydedin:

python pipeline_train.py
Bu komut, tüm ön işleme adımlarını ve meta modeli eğiterek tek bir pickle dosyası halinde kaydeder.

Adım 4: Streamlit Uygulamasını Başlatma
Uygulamayı başlatmak için terminalde aşağıdaki komutu çalıştırın:

streamlit run App.py
Komut çalıştıktan sonra, projeniz tarayıcınızda açılacaktır.

🖥️ Kullanım
Uygulama arayüzü, iki farklı şekilde tahmin yapmanıza olanak tanır:

Bireysel Gösterge Girişi: Formu doldurarak bir kişinin sağlık göstergelerini manuel olarak girip anında tahmin sonucunu görebilirsiniz.

CSV Dosyası ile Toplu Tahmin: Veri setinize uygun formatta bir CSV dosyası yükleyerek, birden fazla kişi için toplu tahmin sonuçları elde edebilirsiniz.

Bu görsele sahip değilseniz, bu satırı silebilir veya kendiniz bir ekran görüntüsü alıp ekleyebilirsiniz.

📈 Sonuçlar
Proje sonucunda, dengesiz veri setinin zorluklarına rağmen özellikle diyabet sınıfının doğru tahmin edilmesinde başarılı sonuçlar elde edilmiştir. Modelin detaylı performans metrikleri (F1 skoru, recall, precision) ve kullanılan tüm yöntemlerin karşılaştırmalı analizi, projenin sunum dosyasında (SugarFlag-Bireysel-Gostergeler-ile-Diyabet-Tahmini (2).pptx) yer almaktadır.

