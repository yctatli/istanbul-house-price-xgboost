# Adım 2 – Keşifsel veri analizi: sonuçlar

Bu dosyadaki sonuçlar `src/analiz.py` çıktılarından alınmıştır:

```bash
./venv/bin/python -m src.analiz
```

`data/processed/temiz_veri.csv` okunur. Çıktılar her çalıştırmada yeniden yazılır.

| Dosya | İçerik |
|---|---|
| `reports/analiz_raporu.txt` | Tüm tablolar ve sayısal bulgular |
| [`plots/01_fiyat_dagilimi.png`](../plots/01_fiyat_dagilimi.png) | Fiyat, log(fiyat) ve m² fiyatı histogramları |
| [`plots/02_m2_fiyat.png`](../plots/02_m2_fiyat.png) | Net m² ve fiyat saçılım grafiği (iki eksen de log) |
| [`plots/03_ilce_m2_fiyat.png`](../plots/03_ilce_m2_fiyat.png) | İlçelere göre medyan m² fiyatı |
| [`plots/04_dogrusal_olmayan.png`](../plots/04_dogrusal_olmayan.png) | Bina yaşı, kat, toplam kat ve m²'nin ilçe içi m² fiyatına etkisi |

> **Göreli m² fiyatı:** Aşağıdaki "ilçe içi" değerlerin hepsi, bir evin m² fiyatının kendi ilçesinin medyanına oranıdır. 1,00 ilçe ortalaması demektir, 1,20 ilçesinden %20 pahalı demektir. Böylece konumun etkisi ayıklanır ve alanın kendi etkisi görülür.

Aşağıdaki sayılar son çalıştırmadan alınmıştır. Temizlik adımı değişirse küçük farklar oluşabilir.

---

## 2.1 Genel bakış

| | |
|---|---|
| İlan sayısı | 24.119 |
| İlçe | 38 |
| Mahalle | 581 |
| Emlak ofisi | 2.395 |
| 5'ten az ilanı olan mahalle | 114 (toplam 233 ilan) |

Az ilanlı mahalleler, `neighborhood` alanında aşırı öğrenme riskinin somut kanıtı. Bu mahalleler modelde gruplanmalı ya da ilçe bilgisiyle yumuşatılmalı.

## 2.2 Hedef değişken: fiyat

| Yüzdelik | Fiyat (TL) |
|---|---|
| %1 | 1.650.000 |
| %25 | 4.500.000 |
| **Medyan** | **6.750.000** |
| %75 | 11.500.000 |
| %99 | 75.000.000 |
| En yüksek | 460.000.000 |

- Ortalama (10,9 M) medyandan (6,75 M) çok yüksek. Az sayıdaki çok pahalı ev ortalamayı yukarı çekiyor.
- Çarpıklık ham fiyatta **8,44**, log fiyatta **0,73**. **Karar:** Model `log(price)` üzerinde eğitilecek.

## 2.3 Sayısal alanlar (Spearman korelasyonu)

| Alan | Fiyatla | İlçe içi m² fiyatıyla | Eksik % |
|---|---|---|---|
| `gross_sqm` | 0,57 | 0,02 | 0 |
| `bathroom_count` | 0,53 | 0,17 | 1,0 |
| `rooms` | 0,52 | −0,02 | 0 |
| `maintenance_fee` | 0,51 | **0,44** | 63,4 |
| `net_sqm` | 0,49 | −0,07 | 0 |
| `floor` | 0,31 | 0,17 | 5,2 |
| `is_in_complex` | 0,24 | 0,34 | 0 |
| `halls` | 0,18 | −0,11 | 0 |
| `building_age` | −0,15 | **−0,39** | 0 |
| `total_floors` | 0,14 | 0,20 | 0 |

- **Büyüklük alanları** toplam fiyatı belirliyor ama m² fiyatına neredeyse hiç etki etmiyor. Yani ev büyüdükçe fiyat artıyor, ama birim fiyat değişmiyor.
- **Kaliteyi gösteren alanlar** m² fiyatını belirliyor: aidat, bina yaşı ve site içinde olmak. Aidat en güçlü kalite göstergesi, ama ilanların %63'ünde eksik.
- **Çoklu doğrusallık doğrulandı:** `net_sqm ~ gross_sqm` r = 0,96, `net_sqm ~ rooms` r = 0,85, `gross_sqm ~ rooms` r = 0,84.

## 2.4 Konum

| | Medyan m² fiyatı (bin TL) |
|---|---|
| En pahalı 3 | Kadıköy 210,8 · Beşiktaş 200,0 · Sarıyer 180,0 |
| En ucuz 3 | Beylikdüzü 44,8 · Sultangazi 43,3 · Esenyurt 32,5 |

- En pahalı ve en ucuz ilçe arasındaki m² fiyatı farkı **6,5 kat**.
- Tek başına ilçe bilgisi, log fiyattaki değişkenliğin **%44**'ünü, log m² fiyatındakinin **%51**'ini açıklıyor.
- Çatalca'da yalnızca 7 ilan var. Bu ilçenin sonuçlarına güvenilmemeli.

Tam tablo `analiz_raporu.txt` dosyasında, grafiği `03_ilce_m2_fiyat.png` dosyasında.

## 2.5 Kategorik alanlar: ham fark ve ilçe içi fark

Ham fark, kategorinin medyan m² fiyatının tüm veriye oranıdır. İlçe içi fark ise konum etkisi ayıklandıktan sonra kalan etkidir. İkisi arasındaki fark büyükse, ham etkinin çoğu konumdan geliyor demektir.

| Alan | Bulgu (ilçe içi) |
|---|---|
| `building_condition` | İnşaat halinde 1,43, yeni 1,15, ikinci el 0,90. Beklendiği gibi güçlü |
| `deed_status` | Arsa tapusu 0,54, hisseli tapu 0,55: çok ucuz. Kat mülkiyeti 1,04, kat irtifakı 0,96 |
| `credit_eligible` | Krediye uygun olmayan 0,73. Beklenen etki doğrulandı |
| `is_in_complex` | Site içi 1,31, değil 0,94 |
| `usage_status` | Boş 1,07, kiracılı 0,89, mülk sahibi oturuyor 0,91 |
| `heating_type` | Merkezi (pay ölçer) 1,40, yerden ısıtma 1,35, kombi 0,93, soba 0,55 |
| `floor_category` | 21 ve üstü 1,38, bodrum 0,67, yarı bodrum 0,62 |
| `exchange` | Takasa açık: ham 0,74, ilçe içi 0,94. **Farkın çoğu konumdan geliyor** |
| `fuel_type` | Elektrik: ham 1,80, ilçe içi 0,87. **Ham fark tamamen konumdan geliyor** |
| `furnished` | Etkisi yok (1,04 / 1,00) |
| `building_type` | Etkisi yok (0,98 – 1,02) |
| `supheli_m2` | İşaretli satırlar 0,82. Net m² gerçekte daha küçük olabilir, bu da m² fiyatını düşük gösteriyor |
| `durum_celiskisi` | 0,95. Zayıf etki |

**Ders:** `exchange` ve `fuel_type` örneklerinde görüldüğü gibi, bir alanın ham etkisine bakarak karar vermek yanıltıcı. Kategorilerin etkisini her zaman ilçe içinde değerlendirmek gerekiyor.

## 2.6 Doğrusal olmayan etkiler

Grafik: `04_dogrusal_olmayan.png`

**Bina yaşı:**

| Yaş | 0 | 1–5 | 6–10 | 11–20 | 21–27 | 28–35 | 36–50 | 50+ |
|---|---|---|---|---|---|---|---|---|
| İlçe içi | 1,17 | 1,10 | 1,00 | 0,94 | **0,69** | **0,72** | 0,85 | 1,15 |

- 20 yaşından sonra keskin bir düşüş var. 27 yaş, 1999 depremi sınırına denk geliyor, beklenen etki doğrulandı.
- 35 yaşından sonra fiyat yeniden artıyor. Büyük ihtimalle bunlar Kadıköy, Beşiktaş gibi merkezi semtlerdeki tarihi ya da kentsel dönüşüm beklenen binalar.

**Kat:** Bodrum 0,81, zemin 0,89, ara katlar ~1,00, 6–10. kat 1,18, 11–20. kat 1,29.

**Binanın kat sayısı:** 6 kata kadar ~0,95, 11–15 kat 1,29, 16–25 kat 1,48. Yüksek binalar yeni siteler demek.

**Kat oranı (kat / toplam kat):**

| Giriş ve altı | Ara katlar | En üst kat |
|---|---|---|
| 0,86 | 1,08 – 1,10 | 0,91 |

En üst kat ara katlardan ucuz. Çatı yalıtımı ve asansör sorunları nedeniyle olabilir. Bu ilişki doğrusal olmadığı için ağaç tabanlı modeller burada avantajlı.

**Net m²:** 50 m² altı 1,28, 75–200 m² arası ~0,95 – 1,00, 300 m² üstü 1,31. Küçük evlerde m² fiyatı yüksek, lüks büyük evlerde de yine yüksek.

## 2.7 Cephe

- Güney cephe ilanların %63,5'inde var, ama fiyat etkisi yok (1,00).
- Cephe sayısının da etkisi yok (0,96 – 1,02).
- **Sonuç:** `orientation` beklenenin aksine bilgi taşımıyor.

## 2.8 Eksiklik bilgi taşıyor mu?

Eksik olan ve dolu olan satırların ilçe içi m² fiyatı karşılaştırıldı:

| Alan | Eksik % | Eksik olanlar | Dolu olanlar |
|---|---|---|---|
| `floor_category` | 70,1 | 1,05 | 0,89 |
| `maintenance_fee` | 63,4 | 1,03 | 0,96 |
| Diğerleri | – | ~1,00 | ~1,00 |

- `floor_category` alanı çoğunlukla bodrum, bahçe katı gibi dezavantajlı katlarda dolduruluyor. Eksikliğinin kendisi bir sinyal.
- Diğer alanlarda eksiklik rastgele görünüyor.

---

## Modelleme için çıkarımlar

1. Hedef `log(price)` olacak.
2. En güçlü etkenler: **net m²** ve **ilçe**. Ardından bina yaşı, aidat, site içi, bina durumu, ısıtma, tapu ve kredi uygunluğu geliyor.
3. `gross_sqm` ile `net_sqm` neredeyse aynı bilgiyi taşıyor (r = 0,96). Doğrusal modelde yalnızca biri kullanılmalı.
4. Yaş, kat ve m²'nin etkileri doğrusal değil. Ağaç tabanlı bir model (XGBoost) doğal seçim. Doğrusal model kullanılırsa bu alanlar gruplanmalı.
5. `kat_orani` (kat / toplam kat) yeni bir özellik olarak eklenecek.
6. `maintenance_fee` ve `floor_category` için "eksik mi" işareti eklenecek.
7. Mahalle, az ilanlı olanlar gruplanarak ya da ilçeye doğru yumuşatılarak kullanılacak.
8. `orientation`, `furnished` ve `building_type` alanlarının katkısı düşük. Eğitimde çıkarılıp çıkarılmayacakları test edilecek.
9. `price_per_sqm` ve `goreli_m2` yalnızca analiz içindir. Fiyattan türetildikleri için **modele girmeyecekler**.
