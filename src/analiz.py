"""
ADIM 2 - KEŞİFSEL VERİ ANALİZİ
Kullanım:  python -m src.analiz
    data/processed/temiz_veri.csv dosyasını okur.
Çıktılar:
    reports/analiz_raporu.txt  -> tüm tablolar ve bulgular
    plots/*.png                -> grafikler
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.yollar import GRAFIK_KLASOR, RAPOR_KLASOR, TEMIZ_VERI

GIRDI = TEMIZ_VERI
CIKTI = GRAFIK_KLASOR
CIKTI.mkdir(exist_ok=True)
RAPOR_KLASOR.mkdir(exist_ok=True)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 20)

rapor_satirlari = []


def rapor(metin=""):
    print(metin)
    rapor_satirlari.append(str(metin))


def baslik(t):
    rapor("\n" + "=" * 75 + f"\n{t}\n" + "=" * 75)


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(CIKTI / ad, dpi=120)
    plt.close()


df = pd.read_csv(GIRDI, encoding="utf-8-sig")
df["log_fiyat"] = np.log(df["price"])
# Göreli m² fiyatı: 1.20 = "ilçesindeki tipik evden %20 pahalı"
df["goreli_m2"] = df["price_per_sqm"] / df.groupby("district")["price_per_sqm"].transform("median")

# ---------------------------------------------------------------- 2.1 Genel
baslik("2.1 GENEL BAKIŞ")
rapor(f"İlan sayısı      : {len(df):,}")
rapor(f"İlçe sayısı      : {df['district'].nunique()}")
rapor(f"Mahalle sayısı   : {df['neighborhood'].nunique()}")
rapor(f"Emlak ofisi sayısı: {df['ofis'].nunique():,}" if "ofis" in df else "")
mah = df.groupby(["district", "neighborhood"]).size()
rapor(f"5'ten az ilanı olan mahalle: {(mah < 5).sum()} / {len(mah)}  "
      f"(bu mahallelerdeki ilanlar: {mah[mah < 5].sum():,})")

# ---------------------------------------------------------------- 2.2 Hedef
baslik("2.2 HEDEF DEĞİŞKEN: FİYAT")
yuzde = [.01, .05, .25, .5, .75, .95, .99]
rapor(df["price"].describe(percentiles=yuzde).apply(lambda x: f"{x:,.0f}").to_string())
rapor(f"\nÇarpıklık (skewness) ham fiyat: {df['price'].skew():.2f}   log fiyat: {df['log_fiyat'].skew():.2f}")
rapor("  (0'a yakın = simetrik. Log dönüşümü dağılımı ne kadar düzeltiyor, buradan görülür)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(df["price"] / 1e6, bins=80)
ax[0].set_title("Fiyat (milyon TL)")
ax[1].hist(df["log_fiyat"], bins=80)
ax[1].set_title("log(Fiyat)")
ax[2].hist(df["price_per_sqm"] / 1e3, bins=80)
ax[2].set_title("m² fiyatı (bin TL)")
kaydet("01_fiyat_dagilimi.png")

# ---------------------------------------------------------------- 2.3 Sayısal alanlar
baslik("2.3 SAYISAL ALANLARIN FİYATLA İLİŞKİSİ (Spearman korelasyonu)")
rapor("Spearman, sıralamaya dayalıdır: doğrusal olmayan ama tek yönlü ilişkileri de yakalar.")
rapor("'fiyat' sütunu: toplam fiyatla ilişki. 'göreli_m2' sütunu: ilçesine göre m² fiyatıyla ilişki.\n")
SAYISAL = ["net_sqm", "gross_sqm", "rooms", "halls", "bathroom_count", "building_age",
           "floor", "total_floors", "maintenance_fee", "is_in_complex"]
SAYISAL = [k for k in SAYISAL if k in df]
kor = pd.DataFrame({
    "fiyat": df[SAYISAL].corrwith(df["log_fiyat"], method="spearman"),
    "göreli_m2": df[SAYISAL].corrwith(df["goreli_m2"], method="spearman"),
    "eksik_%": df[SAYISAL].isna().mean() * 100,
}).sort_values("fiyat", key=abs, ascending=False)
rapor(kor.round(3).to_string())

rapor("\nBirbiriyle güçlü ilişkili alan çiftleri (|r| > 0.7):")
ic = df[SAYISAL].corr(method="spearman")
for i, a in enumerate(SAYISAL):
    for b in SAYISAL[i + 1:]:
        if abs(ic.loc[a, b]) > 0.7:
            rapor(f"  {a:<16} ~ {b:<16} r = {ic.loc[a, b]:.2f}")

plt.figure(figsize=(7, 5))
plt.scatter(df["net_sqm"], df["price"] / 1e6, s=3, alpha=0.2)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Net m² (log)"); plt.ylabel("Fiyat, milyon TL (log)")
plt.title("Metrekare ve fiyat")
kaydet("02_m2_fiyat.png")

# ---------------------------------------------------------------- 2.4 Konum
baslik("2.4 KONUM: İLÇELERE GÖRE")
ilce = df.groupby("district").agg(
    ilan=("price", "size"),
    medyan_fiyat_M=("price", lambda s: s.median() / 1e6),
    medyan_m2_bin=("price_per_sqm", lambda s: s.median() / 1e3),
    medyan_net_m2=("net_sqm", "median"),
).sort_values("medyan_m2_bin", ascending=False)
rapor(ilce.round(1).to_string())
oran = ilce["medyan_m2_bin"].max() / ilce["medyan_m2_bin"].min()
rapor(f"\nEn pahalı / en ucuz ilçe m² fiyatı oranı: {oran:.1f} kat")

ilce["medyan_m2_bin"].sort_values().plot.barh(figsize=(8, 11), title="İlçelere göre medyan m² fiyatı (bin TL)")
kaydet("03_ilce_m2_fiyat.png")

# İlçe içindeki yayılım: ilçe bilgisi fiyatı ne kadar açıklıyor?
toplam_var = df["log_fiyat"].var()
ilce_ici_var = df.groupby("district")["log_fiyat"].var().mul(df.groupby("district").size()).sum() / len(df)
rapor(f"\nSadece ilçe bilinirse log fiyattaki değişkenliğin %{(1 - ilce_ici_var / toplam_var) * 100:.0f}'i açıklanır.")
m2_log = np.log(df["price_per_sqm"])
ilce_ici_m2 = df.assign(l=m2_log).groupby("district")["l"].var().mul(df.groupby("district").size()).sum() / len(df)
rapor(f"Sadece ilçe bilinirse log m² fiyatındaki değişkenliğin %{(1 - ilce_ici_m2 / m2_log.var()) * 100:.0f}'i açıklanır.")

# ---------------------------------------------------------------- 2.5 Kategorik alanlar
baslik("2.5 KATEGORİK ALANLAR: HAM FARK vs İLÇE İÇİ FARK")
rapor("ham_etki   : kategorinin medyan m² fiyatı / tüm verinin medyanı")
rapor("ilçe_içi   : kategorideki evlerin göreli m² fiyatının medyanı (ilçe etkisi ayıklanmış)")
rapor("İkisi çok farklıysa, ham fark büyük ölçüde konumdan kaynaklanıyordur.\n")
genel_medyan = df["price_per_sqm"].median()
KATEGORIK = ["building_condition", "deed_status", "credit_eligible", "is_in_complex", "usage_status",
             "furnished", "heating_type", "fuel_type", "floor_category", "building_type", "exchange",
             "supheli_m2", "durum_celiskisi"]
for kol in [k for k in KATEGORIK if k in df]:
    t = df.groupby(df[kol].fillna("(eksik)").astype(str)).agg(
        ilan=("price", "size"),
        ham_etki=("price_per_sqm", lambda s: s.median() / genel_medyan),
        ilce_ici=("goreli_m2", "median"),
    )
    t = t[t["ilan"] >= 30].sort_values("ilce_ici", ascending=False)
    rapor(f"--- {kol} ---")
    rapor(t.round(2).to_string())
    rapor()

# ---------------------------------------------------------------- 2.6 Doğrusal olmayan etkiler
baslik("2.6 DOĞRUSAL OLMAYAN ETKİLER (ilçe içi göreli m² fiyatı)")
gruplar = {
    "building_age": [-1, 0, 5, 10, 20, 27, 35, 50, 200],
    "floor": [-10, -1, 0, 1, 3, 5, 10, 20, 100],
    "total_floors": [0, 2, 4, 6, 10, 15, 25, 100],
    "net_sqm": [0, 50, 75, 100, 125, 150, 200, 300, 1000],
}
fig, eks = plt.subplots(2, 2, figsize=(13, 8))
for (kol, sinir), ax in zip(gruplar.items(), eks.flat):
    kutu = pd.cut(df[kol], sinir)
    t = df.groupby(kutu, observed=True)["goreli_m2"].agg(["size", "median"])
    rapor(f"--- {kol} ---")
    rapor(t.round(2).to_string())
    rapor()
    ax.bar(t.index.astype(str), t["median"])
    ax.axhline(1, color="red", ls="--", lw=1)
    ax.set_title(f"{kol}: ilçesine göre m² fiyatı")
    ax.tick_params(axis="x", rotation=45)
kaydet("04_dogrusal_olmayan.png")

# Kat oranı: 3 katlı binada 3. kat ile 20 katlı binada 3. kat aynı değil
df["kat_orani"] = df["floor"] / df["total_floors"].replace(0, np.nan)
t = df.groupby(pd.cut(df["kat_orani"], [-1, 0, 0.25, 0.5, 0.75, 0.99, 1.0]), observed=True)["goreli_m2"].agg(["size", "median"])
rapor("--- kat_orani (kat / toplam kat) ---")
rapor(t.round(2).to_string())

# ---------------------------------------------------------------- 2.7 Cephe
baslik("2.7 CEPHE")
yon = df["orientation"].fillna("")
for y in ["North", "South", "East", "West"]:
    var = yon.str.contains(y)
    rapor(f"{y:<6}: {var.mean() * 100:5.1f}% ilanda  | var: {df.loc[var, 'goreli_m2'].median():.2f}  "
          f"yok: {df.loc[~var & (yon != ''), 'goreli_m2'].median():.2f}")
cephe_sayisi = yon.apply(lambda s: len([p for p in s.split(",") if p.strip()]))
t = df[yon != ""].groupby(cephe_sayisi[yon != ""])["goreli_m2"].agg(["size", "median"])
rapor("\nCephe sayısına göre:")
rapor(t.round(2).to_string())

# ---------------------------------------------------------------- 2.8 Eksik değerler bilgi taşıyor mu?
baslik("2.8 EKSİKLİK BİLGİ TAŞIYOR MU?")
rapor("Bir alanın boş bırakılması tesadüf mü, yoksa belli tür evlerde mi oluyor?\n")
for kol in df.columns:
    orani = df[kol].isna().mean()
    if 0.02 < orani < 0.98 and kol not in ("complex_name",):
        eksik = df.loc[df[kol].isna(), "goreli_m2"].median()
        dolu = df.loc[df[kol].notna(), "goreli_m2"].median()
        isaret = "  <-- dikkat" if abs(eksik - dolu) > 0.10 else ""
        rapor(f"{kol:<20} eksik %{orani * 100:5.1f} | eksik olanlar: {eksik:.2f}  dolu olanlar: {dolu:.2f}{isaret}")

# ---------------------------------------------------------------- Kaydet
with open(RAPOR_KLASOR / "analiz_raporu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari))
print(f"\nKaydedildi: {RAPOR_KLASOR}/analiz_raporu.txt ve {CIKTI}/ grafikleri")