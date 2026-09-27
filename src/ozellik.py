"""
ADIM 3 - ÖZELLİK MÜHENDİSLİĞİ
Kullanım:  python -m src.ozellik
    data/processed/temiz_veri.csv dosyasını okur.
Çıktılar (data/features/ klasörüne yazılır, rapor reports/ klasörüne):
    ozellikler.csv       -> modele girecek özellikler + hedef + gruplama anahtarları
    ozellik_listesi.json -> hangi sütun sayısal, hangisi kategorik (4. adım okurken kullanır)
    ozellik_raporu.txt   -> özet

ÖNEMLİ KURAL: Bu adımda hiçbir özellik FİYAT kullanılarak hesaplanmaz.
Fiyattan türetilen her şey (m² fiyatı, mahalle fiyat ortalaması vb.) hedef sızıntısı riski taşır
ve ancak eğitim/test ayrımından SONRA, sadece eğitim verisiyle hesaplanabilir (4. adım).
"""
import json

import numpy as np
import pandas as pd

from src.yollar import OZELLIK_KLASOR, RAPOR_KLASOR, TEMIZ_VERI

GIRDI = TEMIZ_VERI
CIKTI = OZELLIK_KLASOR
CIKTI.mkdir(parents=True, exist_ok=True)
RAPOR_KLASOR.mkdir(exist_ok=True)

MIN_MAHALLE_ILAN = 10   # bundan az ilanı olan mahalleler "İlçe / Diğer" altında toplanır
MIN_KATEGORI_ILAN = 20  # bundan az görülen kategoriler "Diğer" olur

rapor_satirlari = []


def rapor(metin=""):
    print(metin)
    rapor_satirlari.append(str(metin))


df = pd.read_csv(GIRDI, encoding="utf-8-sig")
rapor(f"Girdi: {len(df):,} ilan\n")
X = pd.DataFrame(index=df.index)

# ================================================================ 3.1 Büyüklük
rapor("3.1 Büyüklük özellikleri")
for kol in ["net_sqm", "gross_sqm", "rooms", "halls", "bathroom_count"]:
    X[kol] = df[kol]
X["net_brut_orani"] = df["net_sqm"] / df["gross_sqm"]
# Aynı m²'de çok oda = küçük odalar. Analizde net_sqm ~ rooms r=0.85 idi;
# bu oran, ikisinin birbirinden FARKLI olan bilgisini ayrı bir özellik olarak verir.
X["oda_basina_m2"] = df["net_sqm"] / (df["rooms"] + df["halls"]).replace(0, np.nan)
rapor("  net_sqm, gross_sqm, rooms, halls, bathroom_count, net_brut_orani, oda_basina_m2")

# ================================================================ 3.2 Kat
rapor("\n3.2 Kat özellikleri")
kat = df["floor"].copy()
kat_turu = df["floor_category"].fillna("")
# Temizlikte kurtarılamayan kat bilgileri: analizde gördüğümüz kategoriler
ek_kurtarma = {"21 and Above": 21, "Sub-level 1": -1, "Sub-level 2": -2, "Sub-level 3": -3,
               "Basement & Ground": -1}
n = 0
for tur, deger in ek_kurtarma.items():
    m = kat.isna() & (kat_turu == tur)
    kat[m] = deger
    n += m.sum()
rapor(f"  Kat türünden {n} ek kat değeri kurtarıldı ('Mid Floor' gibi belirsizler boş kaldı)")

X["floor"] = kat
X["total_floors"] = df["total_floors"]
X["kat_orani"] = kat / df["total_floors"].replace(0, np.nan)
# Analiz: en üst kat 0.91, ara katlar ~1.09, bodrum 0.62-0.75
X["en_ust_kat"] = ((kat == df["total_floors"]) | kat_turu.isin(["Top Floor", "Penthouse"])).astype(int)
X["yer_alti"] = (kat < 0).astype(int)
X["giris_kati"] = (kat == 0).astype(int)
X.loc[kat.isna(), ["yer_alti", "giris_kati"]] = np.nan  # bilinmiyorsa "hayır" deme
rapor("  floor, total_floors, kat_orani, en_ust_kat, yer_alti, giris_kati")

# ================================================================ 3.3 Bina
rapor("\n3.3 Bina özellikleri")
X["building_age"] = df["building_age"]
X["is_in_complex"] = df["is_in_complex"]
X["maintenance_fee"] = df["maintenance_fee"]
# Aidat, büyük dairede doğal olarak yüksek. m² başına aidat "lüks seviyesini" daha saf ölçer.
X["aidat_m2"] = df["maintenance_fee"] / df["net_sqm"]
rapor("  building_age, is_in_complex, maintenance_fee, aidat_m2")
rapor("  (Aidat %63 eksik: doldurmuyoruz. XGBoost eksik değerlerin hangi dala gideceğini kendisi öğrenir)")

# ================================================================ 3.4 Cephe
rapor("\n3.4 Cephe (analizde etkisi zayıftı; modelin karar vermesine izin veriyoruz)")
yon = df["orientation"]
for ing, tr in [("North", "kuzey"), ("South", "guney"), ("East", "dogu"), ("West", "bati")]:
    X[f"cephe_{tr}"] = yon.str.contains(ing).astype(float)
    X.loc[yon.isna(), f"cephe_{tr}"] = np.nan
X["cephe_sayisi"] = X[["cephe_kuzey", "cephe_guney", "cephe_dogu", "cephe_bati"]].sum(axis=1, min_count=1)
rapor("  cephe_kuzey, cephe_guney, cephe_dogu, cephe_bati, cephe_sayisi")

# ================================================================ 3.5 İlan
rapor("\n3.5 İlan bilgileri")
gun = (pd.to_datetime(df["scraped_at"]).dt.normalize() - pd.to_datetime(df["last_updated"])).dt.days
X["son_guncelleme_gun"] = gun
rapor(f"  son_guncelleme_gun (ilan en son kaç gün önce güncellenmiş; %{gun.isna().mean() * 100:.0f} eksik)")
X["supheli_m2"] = df["supheli_m2"]
X["durum_celiskisi"] = df["durum_celiskisi"]
rapor("  supheli_m2, durum_celiskisi (temizlikte koyduğumuz işaretler)")

# ================================================================ 3.6 Kategorik alanlar
rapor("\n3.6 Kategorik alanlar")
KATEGORIK_ALANLAR = ["district", "floor_category", "building_condition", "building_type", "deed_status",
                     "credit_eligible", "usage_status", "furnished", "heating_type", "fuel_type", "exchange"]
for kol in KATEGORIK_ALANLAR:
    s = df[kol].fillna("Bilinmiyor")
    sayim = s.value_counts()
    nadir = sayim[sayim < MIN_KATEGORI_ILAN].index
    s = s.where(~s.isin(nadir), "Diğer")
    X[kol] = s
    ek = f"  ({len(nadir)} nadir değer 'Diğer' oldu: {', '.join(map(str, nadir))})" if len(nadir) else ""
    rapor(f"  {kol:<20} {s.nunique():>3} kategori{ek}")

# Mahalle: aynı isim farklı ilçelerde olabilir ("Merkez") -> ilçeyle birleştir
mah = df["district"] + " / " + df["neighborhood"].fillna("Bilinmiyor")
sayim = mah.value_counts()
az = sayim[sayim < MIN_MAHALLE_ILAN].index
mah = mah.where(~mah.isin(az), df["district"] + " / Diğer")
X["mahalle"] = mah
rapor(f"  {'mahalle':<20} {mah.nunique():>3} kategori  ({len(az)} mahalle, "
      f"toplam {sayim[az].sum()} ilan, '<İlçe> / Diğer' altında toplandı)")

KATEGORIK = KATEGORIK_ALANLAR + ["mahalle"]
SAYISAL = [k for k in X.columns if k not in KATEGORIK]

# ================================================================ 3.7 Hedef ve anahtarlar
rapor("\n3.7 Hedef ve gruplama anahtarları (bunlar ÖZELLİK DEĞİL)")
X["log_fiyat"] = np.log(df["price"])
X["price"] = df["price"]
X["listing_id"] = df["listing_id"]
X["ofis"] = df["ofis"].astype(str)
rapor("  log_fiyat (hedef), price (değerlendirme için), listing_id, ofis (eğitim/test ayrımı için)")

# ================================================================ Kontrol ve kayıt
rapor("\n" + "=" * 60)
rapor(f"Toplam özellik: {len(SAYISAL) + len(KATEGORIK)}  ({len(SAYISAL)} sayısal, {len(KATEGORIK)} kategorik)")
rapor("Modele GİRMEYENLER: price_per_sqm (sızıntı), complex_name (is_in_complex yeterli),")
rapor("  orientation (cephe_* sütunlarına bölündü), tarihler (son_guncelleme_gun'e dönüştü), listing_id, ofis")

eksik = (X[SAYISAL].isna().mean() * 100).round(1)
rapor("\nEksik değer oranı (%) - sayısal özellikler:")
rapor(eksik[eksik > 0].sort_values(ascending=False).to_string())

rapor("\nÖrnek: ilk ilanın özellikleri")
rapor(X.iloc[0].to_string())

X.to_csv(CIKTI / "ozellikler.csv", index=False)
with open(CIKTI / "ozellik_listesi.json", "w", encoding="utf-8") as f:
    json.dump({"sayisal": SAYISAL, "kategorik": KATEGORIK, "hedef": "log_fiyat", "grup": "ofis"},
              f, ensure_ascii=False, indent=2)
with open(RAPOR_KLASOR / "ozellik_raporu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari))
print(f"\nKaydedildi: {CIKTI}/ozellikler.csv, ozellik_listesi.json, {RAPOR_KLASOR}/ozellik_raporu.txt")