"""
ADIM 1 - VERİ TEMİZLİĞİ
Kullanım:  python -m src.clear_data [dosya.csv]
    Dosya verilmezse data/raw/ klasöründeki CSV dosyası okunur.
Çıktılar:
    data/processed/temiz_veri.csv          -> sonraki adımlarda kullanılacak temiz veri
    data/processed/cikarilan_satirlar.csv  -> silinen her satır ve silinme nedeni (kontrol için)
    reports/temizlik_raporu.txt     -> her adımda ne yapıldığının özeti
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.yollar import HAM_KLASOR, KOK, RAPOR_KLASOR, TEMIZ_KLASOR

TEMIZ_KLASOR.mkdir(parents=True, exist_ok=True)
RAPOR_KLASOR.mkdir(exist_ok=True)

if len(sys.argv) > 1:
    GIRDI = Path(sys.argv[1])
    if not GIRDI.is_absolute() and not GIRDI.exists():
        GIRDI = HAM_KLASOR / GIRDI
else:
    csvler = sorted(HAM_KLASOR.glob("*.csv"))
    if not csvler:
        sys.exit(f"HATA: {HAM_KLASOR} içinde CSV dosyası bulunamadı")
    if len(csvler) > 1:
        sys.exit("HATA: data/raw içinde birden fazla CSV var, hangisinin kullanılacağını belirtin:\n  "
                 + "\n  ".join(p.name for p in csvler))
    GIRDI = csvler[0]
rapor_satirlari = []
cikarilanlar = []


def rapor(metin):
    print(metin)
    rapor_satirlari.append(metin)


def cikar(df, maske, neden):
    """Maskeye uyan satırları çıkarır ve nedenini kaydeder."""
    silinen = df[maske].copy()
    if len(silinen):
        silinen["cikarilma_nedeni"] = neden
        cikarilanlar.append(silinen)
    rapor(f"  - {neden:<55} {maske.sum():>7,} satır çıkarıldı")
    return df[~maske]


df = pd.read_csv(GIRDI, encoding="utf-8-sig")
BASLANGIC = len(df)
rapor(f"Girdi: {GIRDI.resolve().relative_to(KOK) if GIRDI.resolve().is_relative_to(KOK) else GIRDI.name}")
rapor(f"Ham veri: {BASLANGIC:,} satır, {df.shape[1]} sütun\n")

# ---------------------------------------------------------------- 1.1 Biçim düzeltme
rapor("1.1 Biçim düzeltme")
# Metin sütunlarındaki boşlukları temizle, boş metinleri 'eksik' yap
for kol in df.select_dtypes(include=["object", "string"]).columns:
    df[kol] = df[kol].astype("string").str.strip().replace("", pd.NA)

SAYISAL = ["price", "rooms", "halls", "gross_sqm", "net_sqm", "floor", "total_floors",
           "building_age", "bathroom_count", "maintenance_fee", "is_in_complex"]
for kol in SAYISAL:
    once = df[kol].isna().sum()
    df[kol] = pd.to_numeric(df[kol], errors="coerce")
    bozuk = df[kol].isna().sum() - once
    if bozuk:
        rapor(f"  - {kol}: sayıya çevrilemeyen {bozuk} değer eksik yapıldı")
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")
rapor("  - Metin sütunları kırpıldı, sayısal ve tarih sütunları dönüştürüldü")

# "Not Specified" gibi değerler gerçek bir kategori değil, eksik bilgidir
BILINMIYOR = ["Not Specified", "Unspecified", "Unknown", "Belirtilmemiş", "-"]
n_bilinmiyor = 0
for kol in df.select_dtypes(include=["string"]).columns:
    maske = df[kol].isin(BILINMIYOR)
    n_bilinmiyor += maske.sum()
    df.loc[maske, kol] = pd.NA
rapor(f"  - 'Not Specified' benzeri {n_bilinmiyor} değer eksik yapıldı")

# ---------------------------------------------------------------- 1.2 Zorunlu alanlar
rapor("\n1.2 Zorunlu alanlar (fiyat, m², ilçe olmadan ilan kullanılamaz)")
df = cikar(df, df["price"].isna() | (df["price"] <= 0), "Fiyat eksik veya sıfır")
df = cikar(df, df["net_sqm"].isna() & df["gross_sqm"].isna(), "Net ve brüt m² ikisi de eksik")
df = cikar(df, df["district"].isna(), "İlçe eksik")

# ---------------------------------------------------------------- 1.3 Tekrarlar
rapor("\n1.3 Tekrar eden ilanlar")
df = df.sort_values("last_updated", ascending=False)  # en güncel olan kalsın
df = cikar(df, df["listing_id"].duplicated(), "Aynı listing_id tekrarı")
AYNI_EV = ["district", "neighborhood", "price", "gross_sqm", "net_sqm", "floor", "building_age", "rooms"]
df = cikar(df, df.duplicated(subset=AYNI_EV), "Aynı ev farklı ilan no ile tekrar girilmiş")

# ---------------------------------------------------------------- 1.4 Metrekare
rapor("\n1.4 Metrekare düzeltmeleri")
# Biri eksikse diğerinden tahmin et (tipik net/brüt oranı ~0.80)
oran = (df["net_sqm"] / df["gross_sqm"]).where(lambda s: s.between(0.5, 0.95)).median()
n_net = df["net_sqm"].isna().sum(); n_brut = df["gross_sqm"].isna().sum()
df["net_sqm"] = df["net_sqm"].fillna((df["gross_sqm"] * oran).round())
df["gross_sqm"] = df["gross_sqm"].fillna((df["net_sqm"] / oran).round())
rapor(f"  - Medyan net/brüt oranı: {oran:.2f}. Eksik net m²: {n_net}, eksik brüt m²: {n_brut} dolduruldu")

# Net > brüt ise büyük ihtimalle yer değiştirmiş
ters = df["net_sqm"] > df["gross_sqm"]
df.loc[ters, ["net_sqm", "gross_sqm"]] = df.loc[ters, ["gross_sqm", "net_sqm"]].values
rapor(f"  - Net > brüt olan {ters.sum()} satırda değerler yer değiştirildi")

df = cikar(df, df["net_sqm"] < 20, "Net m² 20'den küçük (hatalı giriş)")
df = cikar(df, df["net_sqm"] > 1000, "Net m² 1000'den büyük (daire için gerçekçi değil)")

# Silmeyip işaretle: net/brüt neredeyse eşit (Heybeliada örneğindeki gibi)
df["supheli_m2"] = ((df["net_sqm"] / df["gross_sqm"]) > 0.95).astype(int)
rapor(f"  - Net/brüt > 0.95 olan {df['supheli_m2'].sum()} satır 'supheli_m2' ile işaretlendi (silinmedi)")

# ---------------------------------------------------------------- 1.5 Kat bilgisi
rapor("\n1.5 Kat bilgisi")
rapor("  - floor_category değerleri:\n" + df["floor_category"].value_counts(dropna=False).to_string())
kat_kat = df["floor_category"].fillna("").str.lower()
eksik_kat = df["floor"].isna()
# En üst kat anlamına gelenler -> toplam kat
ust = eksik_kat & kat_kat.str.contains("top|penthouse|roof") & df["total_floors"].notna()
df.loc[ust, "floor"] = df.loc[ust, "total_floors"]
# Giriş seviyesi anlamına gelenler -> 0
giris = eksik_kat & kat_kat.str.contains("entrance|ground|raised|garden") & ~kat_kat.str.contains("semi|basement")
df.loc[giris, "floor"] = 0
# Bodrum -> -1
bodrum = eksik_kat & kat_kat.str.contains("basement|semi")
df.loc[bodrum, "floor"] = -1
rapor(f"  - Kat eksikti, kat türünden kurtarıldı: üst kat {ust.sum()}, giriş {giris.sum()}, bodrum {bodrum.sum()}")

hatali_kat = df["floor"] > df["total_floors"]
df.loc[hatali_kat, "floor"] = np.nan
rapor(f"  - Kat > toplam kat olan {hatali_kat.sum()} satırda kat eksik yapıldı")

hatali_top = (df["total_floors"] <= 0) | (df["total_floors"] > 80)
df.loc[hatali_top, "total_floors"] = np.nan
rapor(f"  - Toplam kat 0 veya 80'den fazla olan {hatali_top.sum()} satırda eksik yapıldı")

# ---------------------------------------------------------------- 1.5b Gizli tekrarlar
rapor("\n1.5b Gizli tekrarlar (aynı ofis, aynı ev, eksik alanlarla tekrar girilmiş)")
# listing_id "48848-283" -> ofis kodu "48848"
df["ofis"] = df["listing_id"].astype("string").str.split("-").str[0]
df["_dolu"] = df.notna().sum(axis=1)  # en çok bilgi içeren ilan kalsın
df = df.sort_values("_dolu", ascending=False)
ANAHTAR = ["ofis", "district", "neighborhood", "price", "gross_sqm", "rooms"]
silinecek = []
for _, grup in df[df.duplicated(ANAHTAR, keep=False)].groupby(ANAHTAR, dropna=False):
    tutulan = []
    for idx, satir in grup.iterrows():
        tekrar = False
        for t in tutulan:
            # kat ve yaş ya eşit ya da birinde eksikse çelişki yok -> aynı ev
            celismiyor = all(pd.isna(satir[k]) or pd.isna(t[k]) or satir[k] == t[k]
                             for k in ["floor", "building_age"])
            if celismiyor:
                tekrar = True
                break
        if tekrar:
            silinecek.append(idx)
        else:
            tutulan.append(satir)
df = cikar(df, df.index.isin(silinecek), "Gizli tekrar (aynı ofis/ev, çelişmeyen bilgi)")
df = df.drop(columns="_dolu")

# ---------------------------------------------------------------- 1.6 Diğer mantık kontrolleri
rapor("\n1.6 Diğer mantık kontrolleri")
for kol, alt, ust in [("building_age", 0, 150), ("bathroom_count", 0, 10),
                      ("rooms", 0, 15), ("maintenance_fee", 0, 200_000)]:
    hatali = (df[kol] < alt) | (df[kol] > ust)
    df.loc[hatali, kol] = np.nan
    rapor(f"  - {kol}: [{alt}, {ust}] dışındaki {hatali.sum()} değer eksik yapıldı")

# Bina yaşı ile durumu çelişkisi: silmeyip işaretle
celiski = ((df["building_condition"] == "New") & (df["building_age"] > 5)).fillna(False).astype(bool)
df["durum_celiskisi"] = celiski.astype(int)
rapor(f"  - 'New' ama 5 yaşından büyük {celiski.sum()} satır 'durum_celiskisi' ile işaretlendi (silinmedi)")

# ---------------------------------------------------------------- 1.7 Aykırı fiyatlar
rapor("\n1.7 Aykırı fiyatlar (ilçe bazında m² fiyatına göre)")
# m² fiyatını kendimiz yeniden hesaplıyoruz (orijinal sütuna güvenmek yerine)
df["price_per_sqm"] = df["price"] / df["net_sqm"]
log_m2 = np.log(df["price_per_sqm"])


def sinirlar(s):
    if len(s) < 30:  # az ilanlı ilçede tüm İstanbul'un sınırını kullan
        s = log_m2
    q1, q3 = s.quantile([0.25, 0.75])
    return pd.Series({"alt": q1 - 3 * (q3 - q1), "ust": q3 + 3 * (q3 - q1)})


sin = log_m2.groupby(df["district"]).apply(sinirlar).unstack()
alt = df["district"].map(sin["alt"]).astype(float)
ust = df["district"].map(sin["ust"]).astype(float)
df = cikar(df, (log_m2 < alt) | (log_m2 > ust), "m² fiyatı ilçesine göre aşırı uç (3×IQR)")

# ---------------------------------------------------------------- 1.8 Son düzenleme
rapor("\n1.8 Son düzenleme")
df = df.drop(columns=["total_rooms"], errors="ignore")  # rooms + halls ile aynı bilgi
rapor("  - total_rooms çıkarıldı (rooms + halls toplamı)")
rapor("  - price_per_sqm yeniden hesaplandı. UYARI: sadece analiz içindir, modele VERİLMEYECEK (hedef sızıntısı)")
df = df.sort_values(["district", "neighborhood", "price"]).reset_index(drop=True)

# ---------------------------------------------------------------- Özet
rapor("\n" + "=" * 60)
rapor(f"Başlangıç: {BASLANGIC:,} satır  ->  Temiz: {len(df):,} satır "
      f"(%{len(df) / BASLANGIC * 100:.1f} korundu)")
rapor("\nTemiz veride kalan eksik değer oranları (%):")
eksik = (df.isna().mean() * 100).round(1)
rapor(eksik[eksik > 0].sort_values(ascending=False).to_string())

df.to_csv(TEMIZ_KLASOR / "temiz_veri.csv", index=False)
if cikarilanlar:
    pd.concat(cikarilanlar).to_csv(TEMIZ_KLASOR / "cikarilan_satirlar.csv", index=False)
with open(RAPOR_KLASOR / "temizlik_raporu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari))
print(f"\nKaydedildi: {TEMIZ_KLASOR}/temiz_veri.csv, cikarilan_satirlar.csv, {RAPOR_KLASOR}/temizlik_raporu.txt")