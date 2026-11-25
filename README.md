# explainable-credit-risk-prediction
Explainable AI-based Credit Risk Prediction and Decision Support System using Machine Learning, SHAP, and LIME.
# Açıklanabilir Yapay Zekâ Tabanlı Kredi Riski Tahmini ve Karar Destek Sistemi

## Proje Hakkında

Bu proje, makine öğrenmesi ve açıklanabilir yapay zekâ (Explainable Artificial Intelligence - XAI) yöntemlerini kullanarak kredi riskini tahmin eden ve model kararlarını açıklayan bir karar destek sistemi geliştirmeyi amaçlamaktadır.

Geliştirilecek sistem, kredi başvurusunda bulunan kişilerin kredi geri ödeme durumlarına ilişkin risk olasılıklarını tahmin edecek ve bu tahminlerin hangi faktörlerden etkilendiğini açıklayacaktır.

Bunun yanı sıra tahmin edilen risk olasılıkları, belirlenen risk eşikleri ve maliyet analizleri doğrultusunda karar destek mekanizmasına aktarılacaktır.

Projenin temel amacı, kredi risk değerlendirme süreçlerinde tahmin başarısının yanı sıra şeffaflık, yorumlanabilirlik ve güvenilirliği de sağlamaktır.

## Projenin Amaçları

- Kredi geri ödememe riskini tahmin edebilen makine öğrenmesi modelleri geliştirmek.
- Yorumlanabilir ve gelişmiş makine öğrenmesi algoritmalarını karşılaştırmak.
- Modellerin performansını uygun değerlendirme metrikleriyle analiz etmek.
- SHAP ve LIME yöntemleriyle model tahminlerini açıklamak.
- Bireysel kredi başvurularında risk tahminini etkileyen temel faktörleri belirlemek.
- Maliyet duyarlı bir karar destek mekanizması oluşturmak.
- Geliştirilen sistemi kullanıcı dostu bir arayüz üzerinden sunmak.

## Kullanılacak Yöntemler

### 1. Veri Seti ve Keşifsel Veri Analizi

- Açık kaynaklı kredi risk veri setlerinin incelenmesi.
- Veri setinin yapısının ve değişkenlerinin analiz edilmesi.
- Eksik ve aykırı değerlerin incelenmesi.
- Keşifsel veri analizi (EDA) gerçekleştirilmesi.
- Verilerin makine öğrenmesi modellerine uygun şekilde hazırlanması.

### 2. Makine Öğrenmesi Modelleri

Projede aşağıdaki algoritmaların kullanılması planlanmaktadır:

- Lojistik Regresyon
- Random Forest
- XGBoost

Modeller aşağıdaki performans metrikleri üzerinden değerlendirilecektir:

- ROC-AUC
- PR-AUC
- KS İstatistiği
- Gini Katsayısı
- Precision ve Recall
- Karışıklık Matrisi
- Brier Skoru

### 3. Açıklanabilir Yapay Zekâ (XAI)

Model tahminlerinin anlaşılabilir ve yorumlanabilir olmasını sağlamak amacıyla aşağıdaki yöntemler kullanılacaktır:

- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Global Özellik Önem Analizi
- Yerel Tahmin Açıklamaları
- Karşı-Olgusal Açıklamalar (Counterfactual Explanations)

### 4. Karar Destek Sistemi

Model tarafından hesaplanan risk olasılıkları, belirlenen eşik değerleri ve maliyet analizi doğrultusunda üç farklı karar kategorisine dönüştürülecektir:

- **Onay:** Düşük riskli başvurular.
- **Manuel İnceleme:** Orta seviyede risk taşıyan başvurular.
- **Red:** Yüksek riskli başvurular.

Sistem, verilen kararların arkasındaki temel faktörleri de kullanıcıya açıklayacaktır.

### 5. Uygulama Geliştirme

Geliştirilen sistemin kullanıcılarla etkileşim kurabilmesi için Streamlit kullanılarak bir web arayüzü oluşturulması planlanmaktadır.

Uygulamada bulunması planlanan özellikler:

- Kredi başvuru bilgi formu
- Kredi geri ödememe risk olasılığı
- Karar destek önerisi
- Bireysel tahmin açıklamaları
- Global özellik önem grafikleri
- Farklı senaryoları incelemek için What-if analizi

## Kullanılacak Teknolojiler

| Kategori | Teknolojiler |
|---|---|
| Programlama Dili | Python |
| Veri İşleme | Pandas, NumPy |
| Makine Öğrenmesi | Scikit-learn, XGBoost |
| Açıklanabilir Yapay Zekâ | SHAP, LIME |
| Veri Görselleştirme | Matplotlib |
| Uygulama Geliştirme | Streamlit |
| Model Kaydetme | Joblib |
| Versiyon Kontrolü | Git, GitHub |

## Proje Klasör Yapısı

```text
explainable-credit-risk-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_veri_kesifsel_analiz.ipynb
│   ├── 02_veri_on_isleme.ipynb
│   ├── 03_model_egitimi.ipynb
│   └── 04_model_aciklanabilirligi.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── decision_support.py
│
├── app/
│   └── app.py
│
├── models/
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Veri Seti

Projede açık kaynaklı kredi risk veri setlerinden yararlanılması planlanmaktadır.

Değerlendirilecek olası veri setleri:

- UCI German Credit
- UCI Default of Credit Card Clients
- Give Me Some Credit

Kullanılacak veri seti, kapsam ve uygunluk değerlendirmesi sonrasında belirlenecektir.

Veri setlerinin kaynakları, lisansları ve kullanım koşulları proje kapsamında belgelenecektir.

## Kurulum

Projenin yerel bilgisayarda çalıştırılması için aşağıdaki adımlar izlenebilir.

### 1. Repository'yi klonlama

```bash
git clone https://github.com/KULLANICI-ADIN/explainable-credit-risk-prediction.git
```

### 2. Proje dizinine geçme

```bash
cd explainable-credit-risk-prediction
```

### 3. Sanal ortam oluşturma

```bash
python -m venv venv
```

### 4. Sanal ortamı etkinleştirme

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 5. Gerekli kütüphaneleri yükleme

```bash
pip install -r requirements.txt
```

## Proje Geliştirme Durumu

- [ ] GitHub repository oluşturulması
- [ ] Python çalışma ortamının hazırlanması
- [ ] Veri setinin seçilmesi ve temin edilmesi
- [ ] Keşifsel veri analizi
- [ ] Veri ön işleme
- [ ] Makine öğrenmesi modellerinin geliştirilmesi
- [ ] Model performanslarının değerlendirilmesi
- [ ] SHAP ve LIME analizleri
- [ ] Karşı-olgusal açıklamaların geliştirilmesi
- [ ] Maliyet duyarlı karar destek sisteminin oluşturulması
- [ ] Adillik ve etik analizlerinin gerçekleştirilmesi
- [ ] Streamlit uygulamasının geliştirilmesi
- [ ] Final raporu ve sunumun hazırlanması

## Etik İlkeler ve Sınırlılıklar

Kredi risk tahmini, bireylerin finansal durumlarını etkileyebilecek önemli bir uygulama alanıdır. Bu nedenle model şeffaflığı, adillik ve sorumlu yapay zekâ kullanımı proje kapsamında dikkate alınacaktır.

Proje akademik ve eğitim amaçlı geliştirilmektedir. Oluşturulacak tahminler ve karar önerileri gerçek kredi değerlendirme süreçlerinde doğrudan kullanılmak üzere tasarlanmamıştır.

Veri setlerinden kaynaklanabilecek yanlılıklar, model sınırlılıkları ve adillik konuları ayrıca değerlendirilecektir.

## Proje Ekibi

- Ekip Üyesi 1: [Merve Başıbüyük]
- Ekip Üyesi 2: [Meryem Başıbüyük]


