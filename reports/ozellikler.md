# Adım 3 – Özellik mühendisliği: sonuçlar

Bu dosyadaki sonuçlar `src/ozellik.py` çıktılarından alınmıştır:

```bash
./venv/bin/python -m src.ozellik
```

`data/processed/temiz_veri.csv` okunur. Çıktılar her çalıştırmada yeniden yazılır.

| Dosya | İçerik |
|---|---|
| `data/features/ozellikler.csv` | Modele girecek özellikler, hedef ve gruplama anahtarları. Her ilan bir satır |
| `data/features/ozellik_listesi.json` | Hangi sütunun sayısal, hangisinin kategorik olduğu, hedef ve grup sütunu. 4. adım bunu okur |
| `reports/ozellik_raporu.txt` | Her adımın özeti, eksik değer oranları ve örnek bir satır |

> **Temel kural:** Bu adımda hiçbir özellik **fiyat kullanılarak** hesaplanmaz. m² fiyatı ya da mahalle fiyat ortalaması gibi fiyattan türetilen her şey hedef sızıntısı yaratır. Bunlar gerekirse 4. adımda, eğitim/test ayrımından **sonra** ve yalnızca eğitim verisiyle hesaplanır.

Aşağıdaki sayılar son çalıştırmadan alınmıştır.

---

## Özet

| | |
|---|---|
| Girdi | 24.119 ilan |
| Toplam özellik | **37** (25 sayısal, 12 kategorik) |
| Hedef | `log_fiyat` |
| Gruplama anahtarı | `ofis` (eğitim/test ayrımı için) |

## 3.1 Büyüklük

| Özellik | Nasıl hesaplandı | Neden |
|---|---|---|
| `net_sqm`, `gross_sqm`, `rooms`, `halls`, `bathroom_count` | Olduğu gibi | Temel büyüklük bilgisi |
| `net_brut_orani` | net m² / brüt m² | Ortak alan payı. Sitelerde düşük olur |
| `oda_basina_m2` | net m² / (oda + salon) | `net_sqm` ile `rooms` çok ilişkili (r = 0,85). Bu oran ikisinin **farklı** olan bilgisini verir: aynı m²'de çok oda, küçük odalar demek |

## 3.2 Kat

- Kat bilgisi boş olan satırlarda kat türünden **187** ek değer kurtarıldı. Örneğin "21 and Above" → 21, "Sub-level 1" → −1.
- "Mid Floor" gibi belirsiz türler boş bırakıldı.

| Özellik | Anlamı | Analizdeki dayanağı |
|---|---|---|
| `floor`, `total_floors` | Kat ve binanın kat sayısı | Yüksek katlar ve yüksek binalar daha pahalı |
| `kat_orani` | kat / toplam kat | 3 katlı binada 3. kat ile 20 katlı binada 3. kat aynı şey değil |
| `en_ust_kat` | En üst kat, çatı katı ya da penthouse mu | En üst kat 0,91, ara katlar ~1,09 |
| `yer_alti` | Kat < 0 mı | Bodrum katlar 0,62 – 0,75 |
| `giris_kati` | Kat = 0 mı | Giriş katı 0,89 |

Kat bilinmiyorsa `yer_alti` ve `giris_kati` 0 yerine **boş** bırakıldı. Bilmediğimiz şeye "hayır" demiyoruz.

## 3.3 Bina

| Özellik | Not |
|---|---|
| `building_age`, `is_in_complex` | Olduğu gibi |
| `maintenance_fee` | Aylık aidat. %63,4 eksik, **doldurulmadı** |
| `aidat_m2` | aidat / net m². Büyük dairenin aidatı doğal olarak yüksektir. m² başına aidat lüks seviyesini daha saf ölçer |

Eksik aidat değerleri tahminle doldurulmadı. XGBoost eksik değerlerin hangi dala gideceğini kendisi öğrenir. Böylece analizde gördüğümüz "aidatın boş olması da bilgi taşıyor" etkisi (eksik 1,03, dolu 0,96) korunur.

## 3.4 Cephe

`orientation` metni dört ayrı 0/1 sütuna bölündü: `cephe_kuzey`, `cephe_guney`, `cephe_dogu`, `cephe_bati`. Bunlara bir de `cephe_sayisi` eklendi.

Analizde cephenin etkisi yok gibi görünüyordu. Yine de model karar versin diye tutuldu. Eğitimde katkısı düşük çıkarsa çıkarılacak. Cephe bilgisi %21,6 ilanda eksik.

## 3.5 İlan bilgileri

| Özellik | Not |
|---|---|
| `son_guncelleme_gun` | `scraped_at` − `last_updated`: ilan en son kaç gün önce güncellenmiş. %48 eksik |
| `supheli_m2`, `durum_celiskisi` | Temizlikte koyduğumuz işaretler. Analizde `supheli_m2` işaretli satırlar 0,82 çıkmıştı |

## 3.6 Kategorik alanlar

Eksik değerler `Bilinmiyor` kategorisine alındı. **20'den az** görülen değerler `Diğer` altında toplandı.

| Alan | Kategori sayısı | `Diğer` olan nadir değerler |
|---|---|---|
| `district` | 38 | Çatalca |
| `floor_category` | 17 | – |
| `building_condition` | 4 | – |
| `building_type` | 5 | Wooden, Brick, Stone, Log |
| `deed_status` | 10 | Foundation/Association, Bilinmiyor |
| `credit_eligible` | 3 | – |
| `usage_status` | 4 | – |
| `furnished` | 3 | – |
| `heating_type` | 12 | VRV, Heat Pump, Solar, Fireplace |
| `fuel_type` | 4 | Coal-Wood, Fuel Oil |
| `exchange` | 3 | – |
| `mahalle` | 575 | 198 mahalle (toplam 823 ilan) |

**Mahalle:**
- Aynı mahalle adı farklı ilçelerde geçebildiği için (örneğin "Merkez") mahalle, ilçe adıyla birleştirildi: `Adalar / Burgazada`.
- **10'dan az** ilanı olan mahalleler kendi ilçelerinin `<İlçe> / Diğer` grubuna alındı. Bu, analizde gördüğümüz aşırı öğrenme riskine karşı alınan önlemdir.

`floor_category` %70 eksik olduğu için `Bilinmiyor` en büyük kategori oldu. Analizde bu alanın eksikliği bilgi taşıyordu (eksik 1,05, dolu 0,89). `Bilinmiyor` kategorisi bu bilgiyi ayrı bir işarete gerek kalmadan taşıyor.

## 3.7 Hedef ve anahtarlar

Bu sütunlar `ozellikler.csv` dosyasında var, ama **özellik değildir**. Modele girdi olarak verilmeyecekler.

| Sütun | Kullanımı |
|---|---|
| `log_fiyat` | Hedef değişken |
| `price` | Tahminleri TL cinsinden değerlendirmek için |
| `listing_id` | Satırları tanımak için |
| `ofis` | Eğitim/test ayrımı. Aynı ofisin ilanları aynı tarafta kalır, böylece model ofisin fiyatlama alışkanlığını ezberleyemez |

## Modele girmeyen ham alanlar

| Alan | Neden |
|---|---|
| `price_per_sqm` | Fiyattan hesaplanıyor, hedef sızıntısı |
| `complex_name` | %85 eksik. `is_in_complex` yeterli |
| `orientation` | `cephe_*` sütunlarına bölündü |
| `last_updated`, `scraped_at` | `son_guncelleme_gun` özelliğine dönüştü |
| `neighborhood` | `mahalle` özelliğine dönüştü |

## Eksik değerler (sayısal özellikler)

| Özellik | Eksik % |
|---|---|
| `maintenance_fee`, `aidat_m2` | 63,4 |
| `son_guncelleme_gun` | 47,9 |
| `cephe_*` (5 sütun) | 21,6 |
| `floor`, `kat_orani`, `yer_alti`, `giris_kati` | 4,4 |
| `bathroom_count` | 1,0 |

Eksik değerler bilerek boş bırakıldı. XGBoost bunları doğrudan işleyebilir. Doğrusal bir model denenirse önce doldurulmaları gerekir.

---

## Sonraki adım için notlar

1. Eğitim/test ayrımı **`ofis` grubuna göre** yapılacak (`ozellik_listesi.json` → `"grup": "ofis"`).
2. Mahalle fiyat ortalaması gibi fiyattan türetilen özellikler istenirse, ayrımdan **sonra** yalnızca eğitim verisiyle hesaplanacak.
3. Katkısı düşük olabilecek özellikler eğitimde test edilecek: `cephe_*`, `furnished`, `building_type`, `supheli_m2`, `durum_celiskisi`.
