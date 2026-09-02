# ZeroPP: Zero-shot Postprocessing Benchmark

## Ne yapıyoruz
Dondurulmuş zaman serisi temel modellerini (TimesFM-3 basta olmak uzere)
istasyon bazli olasiliksal hava tahmini post-processing'inde, egitimli
EMOS/DRN baseline'larina karsi kiyasliyoruz. Ana soru: kac gunluk egitim
verisinden sonra egitimli yontemler sifir atisli modeli geciyor.

## Veri
EUPPBench (EUMETNET Postprocessing Benchmark) v1.0.
- GitHub: EUPP-benchmark/climetlab-eumetnet-postprocessing-benchmark
- Zenodo DOI: 10.5281/zenodo.7429236
- Hedefler: t2m (birincil), w10 (ikincil)
- EUPPBench'in KENDI train/test bolmesini kullan, yeniden bolme YAPMA.

## Mimari kurallar (pazarlik yok)
1. Her model `src/zeropp/models/base.py` icindeki `Postprocessor` ABC'sini
   implement eder: `fit(train) -> self`, `predict_quantiles(X) -> ndarray`
   sekli (n_samples, n_leads, n_quantiles). Sifir atisli modellerde fit()
   no-op olur ama imza degismez.
2. Kuantil seviyeleri global sabit: [0.1, 0.2, ..., 0.9]. configs/ icinde.
3. Tum sonuclar results/ altina parquet + json. Her dosyaya git SHA,
   model versiyonu ve config hash'i yazilir.
4. Notebook'lar sonuc URETMEZ. Sadece kesif. Sonuc scripts/ ile uretilir.
5. Rastgelelik: her seed configs/experiment.yaml icinde. Kod icinde
   hardcoded seed veya cagri yok.

## TimesFM-3 kovaryat kurgusu
- target: istasyon gozlemi (gecmis penceresi)
- past-future (dynamic) covariates: ensemble ortalamasi, ensemble spread,
  ek NWP alanlari. Bunlar ufuk boyunca BILINEN olarak verilir. Bu projenin
  teknik kalbi burasi, dikkatli implement et ve birim testi yaz.
- Model dondurulmus. Fine-tuning YOK (ayri bir kol olarak eklenene kadar).

## Veri kalitesi kisitlari (TSFM gereksinimleri)
- Context kesintisiz olmali, delik olmamali
- Context ve ufuk ayni frekansta olmali
- NaN'lar model cagrilmadan once lineer interpolasyonla doldurulmali
- DST gecisleri ve UTC/yerel saat karisikligi sahte delik yaratir, qc.py
  bunlari yakalamali ve testleri olmali

## Metrikler (hepsi zorunlu, tek skorla karar verilmez)
CRPS, MAE, pinball loss, PIT histogrami, nominal %90'da ampirik kapsama,
reliability index, threshold-weighted CRPS (kuyruk), wall-clock sure.

## Baseline seti (eksik baseline = reddedilmis makale)
raw ensemble, climatology, persistence, EMOS, AR-EMOS,
time-series EMOS (Jobst 2024), DRN (Rasp & Lerch 2018),
QRF (Taillardat 2016), MOS random forests (Muschinski 2023),
lead-time-continuous (Wessel 2024).
TSFM tarafi: TimesFM-3, Chronos-2, Moirai-2, CITRAS-FM.

## Veri boyutu ekseni
N in {0, 30, 90, 365, 1095, full} gun. Her N icin tum egitimli modeller
yeniden fit edilir, sifir atisli modeller degismez. Cikti: CRPS vs N egrisi.

## Ortam
SSH sunucu, Python 3.11, uv ile yonetilen venv.
GPU olmayabilir, CPU yolunu her zaman calisir tut.
Uzun koşular tmux icinde. SSH kopmasi is oldurmemeli.

## Lisans notu
TimesFM 3.0 agirliklari TimesFM Non-Commercial License v1.0 altinda.
Akademik kullanim uygun. Ticari kullanim veya urune gomme YOK.
README'de bunu belirt.

## Yapma
- EUPPBench bolmesini degistirme
- Sonuclari notebook'ta uretme
- Tek metrikle sonuc iddia etme
- Yeni mimari gelistirme, bu bir uygulama ve benchmark calismasi
- Kovaryat enjeksiyonunu "yaklasik" implement etme, dogru veya hic
