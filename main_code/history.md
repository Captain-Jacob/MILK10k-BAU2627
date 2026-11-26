# MILK10k — Deney Geçmişi

Bu dosya, MILK10k (ISIC 2025) cilt lezyonu sınıflandırma projesinde yapılan bütün deneyleri, sonuçlarını ve kararları tutar. Yeni bir deney yapıldığında en alta eklenir.

## Değerlendirme yöntemi

- **Veri:** 5240 eğitim lezyonu, 11 sınıf. Her lezyon için bir klinik ve bir dermoskopik görüntü ile metadata (yaş, cinsiyet, cilt tonu, lezyon yeri ve 14 MONET skoru) var. Test setinin (479 lezyon) etiketleri gizli.
- **Ölçüm:** Lezyon bazında 5-fold stratified cross-validation (seed 42). Bütün deneylerde aynı fold'lar kullanıldı.
- **Metrik:** Yarışmanın resmi metriği olan **macro-F1**. Her sınıfın F1'inin düz ortalaması alınıyor, yani 9 örnekli MAL_OTH da 2522 örnekli BCC kadar önemli.
- **Kıyaslama:** Her deney o anki referansla aynı fold'lar üzerinde karşılaştırıldı.
  - **Kaç fold'da önde:** Referansı 5 fold'un kaçında geçtiğine bakıldı.
  - **Gürültü seviyesi:** Fold'lar arası sapma ±0.011–0.015. Bundan küçük farklar gürültü olarak değerlendirildi.
- **Kabul kuralı:** Bir değişiklik ancak hem ortalamada anlamlı bir kazanç (yaklaşık ≥ +0.01) sağlıyor hem de fold'ların çoğunda referanstan iyi çıkıyorsa kabul edildi.
- **İç içe CV:** Veriden öğrenilen fusion ağırlığı ve karar eşikleri gibi ayarlar iç içe CV ile ölçüldü. Her fold'un ayarı sadece diğer 4 fold'dan öğrenildi.

## Mevcut durum

| | Pipeline | CV macro-F1 | Test (leaderboard) |
|---|---|---|---|
| **En iyi sistem** (`notebooks/02_finetune.ipynb`) | fine-tuned DINOv2-base + donmuş DINOv2-large (LR) + tabular LightGBM, 1/3'er ortalama | **0.544** (fold sapması 0.006) | **0.511** |
| Donmuş sistem (`notebooks/01_baseline.ipynb`) | DINOv2-large özellikleri + LR (C=0.03) · LightGBM tabular · düz ortalama (`FUSION_W = 0.5`) | 0.487 (fold sapması 0.024) | 0.421 |

Başlangıçtan bu yana kazanç (CV): 0.425 → 0.544 (+0.119).

## Test seti (leaderboard) gönderimleri

| Tarih | Sistem | CV macro-F1 | Test macro-F1 (Dice) | Fark |
|---|---|---|---|---|
| 2026-10-22 | DINOv2-large (donmuş) + LR · LightGBM · düz ortalama | 0.487 | 0.421 | −0.066 |
| 2026-10-23 | fine-tuned DINOv2-base + donmuş DINOv2-large + tabular (1/3'er) | 0.544 | **0.511** | −0.033 |
| 2026-11-23 | aynı üç model; fine-tuned kısım = 5 fold modelinin ortalaması × 8 görünümlü TTA | 0.560 | 0.474 | −0.086 |

Leaderboard referansı (2026-10-22): 1. sıra 0.705, 10. sıra 0.571.

- **Not:** Gönderim dosyasında ilk sütunun adı `lesion_id` olmalı. Yarışma sayfası `lesion` diyor ama resmi skorlama kodu `image` veya `lesion_id` istiyor; ilk gönderim bu yüzden reddedildi.

## Özet tablo

Δ sütunu, deneyin yapıldığı andaki referansa göre farkı gösteriyor.

| # | Tarih | Deney | Görüntü F1 | Fusion F1 | Δ | Önde olduğu fold | Karar |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-15 | Baseline: ResNet50 + LR · LightGBM · ortalama | 0.388 | 0.425 | referans | – | referans |
| 2 | 2026-10-21 | ResNet50 → ResNet101 | 0.395 | 0.436 | +0.011 | 5/5 | ✅ kabul |
| 3 | 2026-10-21 | LR yerine SVM / MLP / LightGBM | 0.335 – 0.381 | 0.343 – 0.411 | −0.022 … −0.095 | 0/5 | ❌ LR kalıyor |
| 4 | 2026-10-21 | class_weight: ağırlıksız / sqrt | 0.376 – 0.404 | 0.351 – 0.419 | −0.019 … −0.087 | 0/5 | ❌ balanced kalıyor |
| 5 | 2026-10-21 | ResNet101 layer2 + layer3 + layer4 birleşimi | 0.427 | 0.452 | +0.016 | 5/5 | ✅ kabul |
| 6 | 2026-10-21 | Fusion: ağırlıklı / geometrik ortalama, stacking | – | 0.371 – 0.456 | ≤ +0.005 | ≤ 3/5 | ➖ düz ortalama kalıyor |
| 7 | 2026-10-21 | Görüntü özellikleri: DINOv2-large | 0.484 | **0.487** | **+0.034** | 5/5 | ✅ kabul (#12'de notebook'a alındı) |
| 8 | 2026-10-22 | Kırpmasız / yüksek çözünürlüklü giriş | 0.412 – 0.425 | 0.431 – 0.437 | −0.017 … −0.021 | 1–2/5 | ❌ kırpma kalıyor |
| 9 | 2026-10-22 | TTA (8 görünüm) | 0.451 | 0.462 | +0.009 | 4/5 | ✅ küçük kazanç |
| 10 | 2026-10-22 | Karar eşikleri (sınıf başına log-bias) | – | 0.438 | −0.016 | 1/5 | ❌ ezber |
| 11 | 2026-10-22 | Tabular özellik mühendisliği | – | 0.432 – 0.435 | −0.018 … −0.020 | 0–1/5 | ❌ |
| 12 | 2026-10-22 | DINOv2-large + fusion ağırlığı ve C'nin birlikte ayarı | 0.484 | 0.488 | ±0.000 (DINOv2 + ortalamaya göre) | 3/5 | ✅ DINOv2 alındı, ➖ düz ortalama kaldı |
| 13 | 2026-10-23 | **Fine-tuning:** DINOv2-base (iki görüş) + donmuş DINOv2-large + tabular | 0.527 | **0.544** | **+0.057** | 5/5 | ✅ kabul · test 0.421 → **0.511** |
| 14 | 2026-11-23 | Fine-tuned'da 8 görünümlü TTA + testte 5 fold modelinin ortalaması (tüm train ile son eğitim yerine) | 0.554 | 0.560 | +0.011 (TTA, aynı modeller) | 4/5 | ❌ test 0.511 → 0.474; 0.511'lik kurulum geri alındı |

---

## Deney ayrıntıları

### 1 · Baseline (2026-10-15)
- **Görüntü:** Klinik ve dermoskopik görüntü dondurulmuş ResNet50'den (ImageNet V1, 256'ya küçültme + ortadan 224 kırpma) geçiriliyor. Görüntü başına 2048, lezyon başına 4096 özellik çıkıyor. Üstünde StandardScaler + Logistic Regression (`class_weight="balanced"`) var.
- **Tabular:** Yaş, cinsiyet, cilt tonu, lezyon yeri ve 14 MONET skoru (toplam 18 özellik) LightGBM'e veriliyor (`class_weight="balanced"`).
- **Fusion:** İki modelin olasılıklarının düz ortalaması alınıp en olası sınıf seçiliyor.
- **Sonuç:** Tabular 0.333, görüntü 0.388, fusion **0.425** (fold sapması 0.011).
- **C seçimi:** LR'nin C değeri 0.0003–0.1 arasında taranıp 0.003 seçildi. C=0.1'de model ezberliyordu ve fusion 0.403'te kalıyordu.
- **Gözlem:** Testte iki model de lezyonların yaklaşık %25'ine SCCKA diyor; train'de bu oran %9. Test setinin sınıf dağılımı muhtemelen farklı.

### 2 · ResNet101 (2026-10-21)
- **Değişiklik:** Sadece backbone değişti.
- **Sonuç:** Görüntü 0.395, fusion **0.436**. Fusion 5 fold'un 5'inde daha iyi, fold ortalaması farkı +0.010.
- **C kontrolü:** C=0.001 ve 0.01 de denendi; ikisi de fusion'da 0.429 verdi, 0.003 en iyisi kaldı.
- **Not:** Macro AUC 0.859'dan 0.841'e düştü. Sınıf kararları iyileşti ama olasılıkların sıralama kalitesi biraz bozuldu.

### 3 · LR yerine başka sınıflandırıcı (2026-10-21)
Aynı ResNet101 özellikleri kullanıldı.

| Model | Görüntü | Fusion |
|---|---|---|
| **LR (C=0.003)** | **0.395** | **0.436** |
| MLP, 20 epoch | 0.381 | 0.411 |
| SVM-RBF (PCA 256, C=10) | 0.379 | 0.389 |
| SVM-RBF (PCA 256, C=1) | 0.381 | 0.382 |
| MLP, 50 epoch | 0.362 | 0.381 |
| LightGBM | 0.335 | 0.343 |

- **Ders:** Donmuş özelliklerin üstünde doğrusal model en iyisi. Esnek modeller ~4200 örneği ezberliyor; MLP eğitimi uzadıkça kötüleşti. SVM'in olasılıkları dengeli olmadığı için fusion'a katkı vermedi.

### 4 · class_weight (2026-10-21)
Her iki model için ağırlıksız, sqrt ve balanced seçenekleri denendi (3×3 kombinasyon).

- **En iyi:** balanced/balanced, 0.436.
- **Diğerleri:** balanced/sqrt 0.419, ağırlıksız/ağırlıksız **0.351**.
- **Ağırlıksız model ne yaptı:** BCC'yi 3336 kez tahmin etti (gerçekte 2522), DF'yi 5 kez (gerçekte 52), VASC'ı 14 kez (gerçekte 47). Precision yükseldi (0.56), recall çöktü (0.32).
- **Nüans:** Görüntü modeli tek başına değerlendirildiğinde sqrt biraz daha iyiydi (0.404'e karşı 0.395). Ama fusion'da balanced kazanıyor.

### 5 · Çok katmanlı ResNet101 özellikleri (2026-10-21)
Her katmanın çıkışı global average pooling ile vektöre indirildi. Her sette C ∈ {0.001, 0.003, 0.01} denendi.

| Özellik seti | Boyut | Görüntü | Fusion |
|---|---|---|---|
| layer4 (önceki) | 4096 | 0.395 | 0.436 |
| layer3 | 2048 | 0.412 | 0.433 |
| layer2 | 1024 | 0.357 | 0.403 |
| layer3 + layer4 | 6144 | 0.417 | 0.435 |
| **layer2 + layer3 + layer4** (C=0.001) | 7168 | **0.427** | **0.452** |

- **Ders:** Orta katmanlardaki doku bilgisi lezyonlar için faydalı; layer3 tek başına bile görüntü modelinde layer4'ü geçti.
- **Not:** Sonuç 12 kombinasyon arasından seçildiği için biraz iyimser olabilir. Ama C=0.003'te de kazanç +0.013 ve 4/5 fold.

### 6 · Fusion yöntemleri (2026-10-21)
İç içe CV ile ölçüldü.

| Yöntem | Fusion | Δ | Önde olduğu fold |
|---|---|---|---|
| Düz ortalama (referans) | 0.452 | – | – |
| Ağırlıklı ortalama (her fold w=0.55 seçti) | 0.456 | +0.005 | 3/5 |
| Geometrik ortalama (w iç içe CV ile seçildi) | 0.439 | −0.015 | 1/5 |
| Geometrik ortalama (w=0.5) | 0.427 | −0.028 | 0/5 |
| Stacking, ağırlıksız hakem LR (C=0.1) | 0.434 | −0.022 | 1/5 |
| Stacking, balanced hakem LR | 0.371 – 0.376 | −0.076 … −0.081 | 0/5 |

- **Geometrik ortalama:** Zayıf tabular modelin "kesinlikle hayır" demesi, görüntü modelinin doğru tahminlerini siliyor.
- **Stacking:** Hakem modelin öğrenmesi için yeterli veri yok. Balanced hakem nadir sınıfları ikinci kez öne çekti; ağırlıksız hakem ise BCC'ye kaydı.
- **Ağırlık duyarlılığı:** w 0.5 ile 0.75 arasında nereye konursa konsun skor yaklaşık 0.45 kalıyor.

### 7 · Daha uygun hazır model: DINOv2 ve MONET (2026-10-21)
Özellik olarak CLS token ile patch token'ların ortalaması birleştirildi. Görüntü başına özellik sayısı DINOv2-base'de 1536, DINOv2-large ve MONET'te 2048.

| Özellik | En iyi C | Görüntü | Fusion | Δ | Önde olduğu fold |
|---|---|---|---|---|---|
| **DINOv2-large** | 0.03 | **0.484** | **0.487** | **+0.034** | 5/5 |
| DINOv2-base | 0.01 | 0.470 | 0.467 | +0.014 | 3/5 |
| MONET (dermatoloji CLIP, `suinleelab/monet`) | 0.003 | 0.448 | 0.457 | +0.005 | 3/5 |
| ResNet101 katmanları + DINOv2-large | 0.01 | 0.479 | 0.484 | +0.030 | 4/5 |

- **Ders:** Kendi kendine öğrenmeyle eğitilmiş DINOv2, ImageNet sınıflandırma modellerinden daha iyi aktarılıyor. ResNet'i DINOv2'nin yanına eklemek bir şey kazandırmadı.
- **MONET neden zayıf kaldı:** Dermatolojiye özel olmasına rağmen DINOv2'nin gerisinde kaldı. Tahminimiz, metinle eşleştirilerek eğitildiği için ince doku detayını taşımaması; bu test edilmedi.
- **Teknik not:** MONET'in ön işlemesi CLIP değil ImageNet ortalama/std değerlerini kullanıyor.
- **Açık nokta:** Görüntü modeli güçlenince fusion ağırlığının yeniden ayarlanması gerekiyor. Ön denemede w=0.6 ve C=0.01 ile 0.494 görüldü; bu henüz iç içe CV ile ölçülmedi.

### 8 · Kırpmasız / yüksek çözünürlüklü giriş (2026-10-22)
ResNet101 layer2+3+4 ile denendi. Görüntünün tamamı en-boy oranı korunarak verildi.

| Giriş | Görüntü | Fusion | Δ |
|---|---|---|---|
| 224×299 | 0.425 | 0.431 | −0.021 |
| 336×448 | 0.419 | 0.436 | −0.017 |
| 450×600 (orijinal) | 0.412 | 0.437 | −0.017 |

- **Ders:** Ortadan kırpmak faydalı. Muhtemelen kenarlardaki gürültüyü (karanlık köşeler, cetvel) atıyor ve lezyonu kadrajda büyütüyor.

### 9 · TTA (2026-10-22)
- **Yöntem:** Referans girişin (256'ya küçültme + ortadan 224 kırpma) 8 görünümü kullanıldı: 4 rotasyon × aynalama. Özellikler bu 8 görünümün ortalaması. Eğitim ve test özelliklerinin ikisine de uygulandı.
- **Sonuç:** C=0.01'de görüntü 0.451, fusion **0.462** (+0.009, 4/5 fold). C=0.03'te 0.456.
- **Maliyet:** Özellik çıkarma süresi yaklaşık 8 katına çıkıyor.

### 10 · Karar eşikleri (2026-10-22)
- **Yöntem:** Fusion olasılıklarına sınıf başına bir log-bias eklenip `argmax(log p + b)` alındı. b, macro-F1'e göre koordinat koordinat arandı.
- **Sonuç:** İç içe CV ile **0.438** (−0.016, 1/5 fold). Bias aynı veride öğrenilip aynı veride ölçülünce 0.472 görünüyordu.
- **Ders:** Nadir sınıflarda birkaç örneğe göre ayar yapınca ezber oluyor; öğrenilen bias'lar fold'dan fold'a çok değişti. MAL_OTH 11 kez tahmin edilmeye başladı, 11'i de yanlıştı. İç içe CV olmasa bu ayar kazanç sanılıp alınacaktı.

### 11 · Tabular özellik mühendisliği (2026-10-22)
- **Yöntem:** Her MONET kavramı için dermoskopik − klinik farkı eklendi (+7 özellik). İkinci varyantta iki görüntünün ortalaması da eklendi (+7 özellik daha).
- **Sonuç:** Tabular F1 0.333'ten 0.326'ya, fusion 0.432–0.435'e düştü (−0.018 … −0.020).
- **Ders:** Birbirine benzeyen ek özellikler sinyal getirmedi, sadece gürültü ekledi.

### 12 · DINOv2-large'ın notebook'a alınması + fusion ağırlığı (2026-10-22)
- **Ayar:** C ∈ {0.01, 0.03, 0.1} ve görüntü ağırlığı w ∈ {0, 0.05, …, 1} iç içe CV ile birlikte seçildi.
- **Seçimler:** Her fold görüntüye biraz daha fazla ağırlık seçti: (C, w) = (0.01, 0.6), (0.01, 0.65), (0.03, 0.55), (0.01, 0.6), (0.03, 0.55).
- **Sonuç:** Ayarlı fusion **0.488**, düz ortalama (C=0.03, w=0.5) **0.487**. Fark yok; ayarlı sürüm 5 fold'un 3'ünde önde.
- **Ders:** Ağırlığı seçtiğimiz veride ölçünce 0.494 (w=0.6, C=0.01) görünüyordu. Bu, karar eşiklerindeki (#10) gibi iyimser bir tepe.
- **Karar:** Notebook'a DINOv2-large, C=0.03 ve düz ortalama ile alındı. Notebook sonucu deneyle birebir aynı (0.487).
- **Diğer ölçümler:** Macro AUC 0.856'dan 0.910'a çıktı. Sınıf bazında en büyük artışlar DF (0.38 → 0.51), BCC (0.81 → 0.86), SCCKA (0.59 → 0.65), BKL (0.37 → 0.45).
- **Gözlem:** Tabular modelin katkısı küçüldü. Görüntü tek başına 0.484, fusion 0.487; fusion'ın fold sapması (0.024) görüntü modelininkinden (0.008) yüksek.
- **Teknik not:** DINOv2 özellikleri yeniden çıkarıldığında cache'ten %0.04–0.17 kadar farklı çıkabiliyor (cosine 0.999999). Sebep GPU'daki TF32 hesaplama ve farklı batch boyutu.

### 13 · Fine-tuning (2026-10-22 → 2026-10-23)
- **Model:** DINOv2-base (ViT-B/14). Klinik ve dermoskopik görüntü aynı backbone'dan geçiyor. Her görüntüden CLS + patch ortalaması alınıyor, ikisi yan yana konup tek bir doğrusal katmana veriliyor.
- **Eğitim:**
  - 10 epoch, AdamW (backbone 1e-5, yeni katman 1e-3), weight decay 0.05.
  - 1 epoch warmup + kosinüs azalan öğrenme hızı.
  - Balanced cross-entropy, fp16, gradient clipping.
  - 6 GB GPU için gradient checkpointing.
  - Her fold'da son epoch kullanıldı; epoch seçimi yapılmadı.
- **Augmentation:** RandomResizedCrop(224, ölçek 0.35–1), yatay ve dikey aynalama, 90° döndürme, ColorJitter. Görüntüler önceden kısa kenar 256 px olacak şekilde küçültüldü.
- **Ana sistem:** fine-tuned + donmuş + tabular, 1/3'er ortalama. Fold 0 pilotunda seçilip diğer fold'lar görülmeden sabitlendi.
- **Süre:** RTX 4050'de fold başına ~30 dk, tüm veriyle son eğitim ~38 dk.

| Sistem (5-fold CV) | Macro-F1 | Fold sapması | 01'e göre | 01'i geçtiği fold |
|---|---|---|---|---|
| 01: donmuş + tabular | 0.487 | 0.024 | – | – |
| Fine-tuned tek başına | 0.527 | 0.015 | +0.038 | 5/5 |
| Fine-tuned + tabular | 0.520 | 0.017 | +0.032 | 5/5 |
| Fine-tuned + donmuş | 0.545 | 0.013 | +0.055 | 5/5 |
| **Fine-tuned + donmuş + tabular (ana sistem)** | **0.544** | **0.006** | **+0.052** | **5/5** |

- **Test (leaderboard):** 0.421'den **0.511**'e çıktı (+0.090). CV ile test arasındaki fark −0.066'dan −0.033'e indi.
- **Sınıf bazında (01 → ana sistem):**
  - En büyük artışlar: AKIEC 0.42 → 0.54, BKL 0.45 → 0.57, VASC 0.69 → 0.80, SCCKA 0.65 → 0.72, DF 0.51 → 0.60.
  - Gerileyen tek sınıf: BEN_OTH 0.25 → 0.21. MAL_OTH hâlâ 0.
- **Ders:**
  - Fine-tuned model tek başına bile 01'in iki modelli sisteminden iyi.
  - Fine-tuned ve donmuş model birbirini tamamlıyor; ensemble'ları ikisinden de iyi.
  - Tabular model ortalamaya neredeyse hiç katkı vermiyor (0.545'e karşı 0.544). Ama fold sapmasını yarıya indiriyor (0.013 → 0.006), yani sonucu daha kararlı yapıyor.
- **Not:** Fold modellerinin ve son modelin ağırlıkları kaydedilmedi, sadece tahminleri kaydedildi. TTA veya fold ensemble için modellerin yeniden eğitilmesi gerekir.

### 14 · TTA + 5 fold ensemble (2026-11-23)
- **Değişiklik:** `common.py`'ye taşıma (davranış aynı), fold ağırlıklarının kaydı (`fold{k}.pt`), fine-tuned tahminlerde 8 görünümlü TTA (4 rotasyon × ayna, iki görüntüye aynı dönüşüm). Testte tüm train ile son eğitim **kaldırıldı**; fine-tuned kısım 5 fold modelinin test tahminlerinin ortalaması oldu.
- **Ortam:** 5 fold yeniden eğitildi (RTX 5070, gradient checkpointing kapalı, fold başına ~6 dk). Ayarlar #13 ile aynı. DINOv2-large özellik cache'i de yeniden çıkarıldı, bu yüzden donmuş LR 0.4837 yerine 0.4821 (#12'deki TF32 notu).

| Sistem (5-fold OOF) | #13 | #14 |
|---|---|---|
| Donmuş DINOv2-large + LR | 0.484 | 0.482 |
| LightGBM | 0.333 | 0.333 |
| Fine-tuned, TTA'sız | 0.527 | 0.535 |
| Fine-tuned, TTA | – | 0.554 |
| Üçlü, TTA'sız | 0.544 | 0.549 |
| Üçlü, TTA | – | **0.560** (fold sapması 0.029) |

- **Test (leaderboard):** **0.474**, 0.511'den −0.037. CV düşmedi, aksine her satır #13'e eşit ya da yukarıda. Gerileme CV'nin ölçmediği yerde: testteki fine-tuned model.
- **Teşhis (cache'teki tahminlerle, eğitim yapmadan):**
  - Kodda hata bulunmadı. Sınıf sırası, lezyon hizalaması, TTA boyutları (`(B, C, H, W)`, 224 ortadan kırpma, normalizasyon sonrası döndürme) doğru. Gönderilen `test_probabilities.csv` cache'ten birebir yeniden üretildi (max fark 1e-16).
  - CV'de her lezyonu **tek** bir fold modeli tahmin ediyor. Testte ise 5 modelin ortalaması kullanıldı; bu kurulumun CV karşılığı yok.
  - Test tahminleri 0.511'lik gönderimden 479 lezyonun 48'inde (%10) farklı. Nadir sınıf tahminleri azaldı: BEN_OTH 1 → 0, BEN_OTH/DF/INF/MAL_OTH/VASC toplamı 23 → 18, BKL 91 → 73. Tek tek fold modelleri BEN_OTH'a 0–2 kez diyor, ortalamaları hiç demiyor. Testte ~4 BEN_OTH olduğu düşünülürse tek bir doğru BEN_OTH tahmini macro-F1'de ~0.03 eder.
  - Aynı ayarlarla eğitilmiş iki fold modeli test lezyonlarının sadece %88'inde aynı sınıfı söylüyor. Yani tek bir eğitim koşusu bile testte birkaç puan oynatabilir.
  - **Leaderboard gürültüsü:** OOF tahminlerinden rastgele 479 lezyon seçilince macro-F1'in sapması **0.045**, aynı alt kümede iki sistemin farkının sapması ~0.02. −0.037'nin bir kısmı gürültü olabilir; leaderboard ±0.01'lik değişiklikleri ayırt edemez.
- **Karar:** Varsayılan test kurulumu 0.511'linkine döndü: `TEST_MODEL = "full"` (tüm train ile tek model, `full.pt`), `TTA = False`. Diğer kombinasyonlar `outputs/finetune/variants/` altına yazılıyor. 0.511 alan orijinal dosya `variants/submission_original_lb0511.csv` (git `f3a0bd61`'den).
- **Aynı gün iç içe CV ile (cache'teki OOF üzerinde) denenenler:**

| Yöntem | Üçlü TTA'sız (0.549) | Üçlü TTA (0.560) |
|---|---|---|
| Öğrenilmiş karışım ağırlıkları (simpleks, 0.1 adım) | +0.003, 3/5 | +0.009, 4/5 (her fold ≈ 0.4 / 0.3 / 0.3 seçti) |
| Tek parametreli öncül düzeltmesi `p / prior^τ` | −0.002, 2/5 | +0.006, 3/5 |
| Ortak sıcaklık (ortalamadan önce `p^(1/T)`) | +0.008, 4/5 (T=3) | −0.010, 0/5 |
| Sınıf başına log-bias (#10 tekrarı) | +0.002, 3/5 (aynı veride 0.566) | −0.008, 1/5 (aynı veride 0.573) |

  Hiçbiri kabul kuralını (≥ +0.01 ve fold'ların çoğu) geçmiyor. Genel ders 1 hâlâ geçerli: üst katman ayarları tıkalı, kazanç görüntü modelinden gelmeli.

---

## Genel dersler
1. **Kazanç hep görüntü özelliklerinden geldi.** ResNet101, çok katmanlı özellikler ve DINOv2 skoru yükseltti. Üst katmandaki ayarlar (sınıflandırıcı, class weight, fusion yöntemi, karar eşikleri) yaklaşık 0.45'te tıkandı.
2. **`class_weight="balanced"` macro-F1 için şart.** Ağırlıksız eğitim fusion skorunu 0.085 düşürdü.
3. **Veriden öğrenilen her ayar iç içe CV ile ölçülmeli.** Karar eşiklerinde aynı veride ölçülen skor ile gerçek skor arasında 0.034 fark çıktı.
4. **MAL_OTH (9 örnek) hiçbir yöntemle öğrenilemedi.** Her sınıf skorun 1/11'i olduğu için pratikteki tavan yaklaşık 0.91.
5. **Test tahmini CV'de ölçülen modelle aynı olmalı ya da leaderboard'da ayrıca doğrulanmalı.** #14'te CV +0.016 yükseldi ama test 0.037 düştü: testte kullanılan 5 model ortalamasının CV karşılığı yoktu. 479 lezyonluk leaderboard'un gürültüsü (σ ≈ 0.045) küçük farkları ayırt edemez.

## Açık kararlar / sıradakiler
- [x] DINOv2-large notebook'a alındı (0.452 → 0.487).
- [x] DINOv2 ile fusion ağırlığı iç içe CV ile yeniden ayarlandı → kazanç yok, düz ortalama kaldı.
- [x] Fine-tuning (DINOv2-base) → CV 0.544, test 0.511.
- [ ] Leaderboard ilk 10 için (0.571) testte yaklaşık +0.06 gerekiyor. Adaylar:
  - ~~Fine-tuned modellerde TTA ve fold ensemble~~ → #14: CV 0.560 ama test 0.474. TTA'yı tek başına (`variants/submission_full_tta.csv`) leaderboard'da denemek açık.
  - Farklı bir fine-tuned backbone ile çeşitlilik.
  - Daha yüksek çözünürlük.
  - Nadir sınıflar için yöntemler.
  - Dış veri (yarışma kurallarına bakılmalı).
- [ ] Tabular modelin katkısı çok küçük. Kararlılık için tutuluyor; güçlendirilmesi düşük öncelikli.

---

## 15 · Temizlik, MCDM değerlendirmesinin doğrulanması ve kilitlenen sonuçlar (2026-11-26)
Yeni deney yok; model eğitilmedi.

- **Temizlik:**
  - Eski EfficientNet-B0 hattının `checkpoints/` altındaki 5 ağırlık dosyası çalışma klasöründen silindi (git geçmişinde `91f9cf46` commit'inde duruyor).
  - `01_baseline.ipynb` ve `evaluate_models.py`'deki katman katman eski ResNet cache okuma yolu kaldırıldı.
  - `requirements.txt`'ye `joblib`, `huggingface_hub`, `matplotlib` eklendi.
  - İki notebook'un kayıtlı çıktılarındaki Windows kullanıcı adı `user` ile değiştirildi (6 yer; kod hücreleri aynı). Yerel `D:\…` yolları duruyor.
- **Doğrulama:** `evaluate_models.py` depo kökünden, mevcut cache'lerle çalıştırıldı (RTX 5070, 106 sn, `HF_HUB_OFFLINE=1`, `--allow-download` olmadan). Aynı günkü 11:09 koşusuyla karşılaştırıldı.
  - **Birebir aynı:** `outputs/mcdm/`'deki 52 dosya (`criteria`, `per_class_metrics`, `fold_metrics`, `model_complexity`, `component_sizes`, 36 confusion matrix dosyası, 9 OOF dosyası, iki `.md`). `decision_matrix.csv`'de beş sınıflandırma metriği, boyut ve parametre sayısı aynı. (`.md` dosyaları script tarafından üretilmiyor; koşudan sonra elle güncellendi.)
  - **Değişen:** sadece latency (`latency_summary.csv`, `latency_stages.csv`, karar matrisindeki süre sütunu) ve `run_info.json`'daki zaman damgası.
  - OOF dosyalarından yeniden hesaplanan pooled ve fold metrikleri tablolarla uyuşuyor (en büyük fark 1e-16); confusion matrix'ler aynı.
  - `outputs/finetune/` ve `outputs/baseline/` değişmedi. `submission.csv` hâlâ `variants/submission_original_lb0511.csv` ile bayt bayt aynı. `02_finetune`'un son eğitim hücresi çalıştırılmadı (`full.pt` yok). Notebook'lar temizlikten sonra yeniden çalıştırılmadı.
- **Cache kontrolleri** (`run_info.json`; değerler önceki koşuyla aynı):

| Kontrol | Sonuç |
|---|---|
| `fold{k}.pt` ile yeniden tahmin ↔ `fold{k}.npz` (fold başına 64 lezyon) | max fark 0.0 |
| `cv_results.csv` ↔ yeniden hesaplanan macro-F1 (4 sistem) | fark 0 (üçlü TTA da aynı: 0.5598) |
| LightGBM yeniden fit ↔ `reference.npz` | max fark 0.0 |
| Donmuş LR yeniden fit ↔ `reference.npz` | **tam aynı değil:** 11 lezyonda farklı sınıf (%99.8 uyum), max olasılık farkı 0.033; macro-F1 0.4822 ↔ 0.4821. Metrikler cache'ten. |
| Ham görüntüden zamanlanan hat ↔ OOF (200 lezyon) | **tam aynı değil:** fine-tuned %96.5 (TTA %97), ResNet101 %99.5, diğerleri %100. Fine-tuned farkının sebebi `img256` JPEG yeniden kaydı (`TECHNICAL_NOTES_FOR_TEAM.md` bölüm 4). |

- **Kilitlenen sonuçlar** (5240 lezyon, OOF; süre = lezyon başına ortalama ms):

| `model_id` | Macro-F1 | Accuracy | Balanced acc. | Weighted F1 | Macro AUC | ms (önceki → son) | MB |
|---|--:|--:|--:|--:|--:|--:|--:|
| `tabular_lgbm` | 0.3334 | 0.6120 | 0.3265 | 0.6044 | 0.8423 | 2.8 → 2.8 | 5.8 |
| `frozen_resnet50_logreg` | 0.4274 | 0.6435 | 0.4658 | 0.6571 | 0.8635 | 11.8 → 11.4 | 103.0 |
| `frozen_resnet101_logreg` | 0.4269 | 0.6391 | 0.4641 | 0.6517 | 0.8483 | 16.7 → 16.1 | 179.3 |
| `frozen_dinov2l_logreg` | 0.4821 | 0.7082 | 0.4732 | 0.7086 | 0.9080 | 33.3 → 33.0 | 1217.8 |
| `late_fusion` | 0.4867 | 0.7273 | 0.4615 | 0.7190 | 0.9099 | 36.2 → 35.8 | 1223.6 |
| `ft_dinov2b` | 0.5350 | 0.7578 | 0.5379 | 0.7581 | 0.9088 | 16.0 → 15.6 | 346.5 |
| `ft_dinov2b_tta` | 0.5543 | 0.7687 | 0.5479 | 0.7678 | 0.9146 | 60.5 → 58.4 | 346.5 |
| `ensemble` | 0.5492 | 0.7781 | 0.5244 | 0.7712 | 0.9304 | 44.9 → 44.4 | 1570.1 |
| `ensemble_tta` | 0.5598 | 0.7821 | 0.5308 | 0.7746 | 0.9322 | 89.3 → 87.2 | 1570.1 |

- **Leaderboard:** en iyi skor hâlâ **0.511** (#13). Bu koşu leaderboard'u ölçmez; #13'ün ağırlıkları kaydedilmediği için 0.544 de yeniden hesaplanamaz.
- **Önceki kayıtlara notlar:**
  - #14'teki "%88 uyum", fine-tuned kısmı farklı fold modelinden gelen üçlü ensemble'lar (TTA'sız) için doğru (ikili ortalama 0.880). Fine-tuned modeller tek başına %79 uyuşuyor.
  - #14'teki "fold başına ~6 dk" yerine, kayıtlı notebook çıktısında 5 fold toplam 35 dk (fold başına ~7 dk).
