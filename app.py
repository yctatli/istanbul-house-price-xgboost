"""
EV FİYAT TAHMİNİ - WEB ARAYÜZÜ
Kullanım:  ./venv/bin/streamlit run app.py
    Tarayıcıda http://localhost:8501 açılır.

Model ve özellik üretimi src/tahmin.py'den gelir (ozellik_uret, tahmin_ve_aciklama, KATEGORI_LISTESI).
Burada hiçbir özellik yeniden hesaplanmaz; arayüz sadece ham ilan bilgisini toplar.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.yollar import TEMIZ_VERI

MEDYAN_HATA = 0.13  # reports/iyilestirme_raporu.txt, test seti
BILMIYORUM = "Bilmiyorum"
AZ_VERI = 10  # ozellik.py: bundan az ilanlı mahalleler "<İlçe> / Diğer" altında toplanır

# ================================================================ Türkçe etiketler
# Modele İngilizce değer gider; ekranda Türkçesi görünür. Emin olunmayanlarda İngilizcesi parantezde.
ETIKET = {
    # ortak
    "Diğer": "Diğer", "Bilinmiyor": "Bilinmiyor",
    # tapu
    "Condominium Title": "Kat mülkiyeti", "Construction Easement": "Kat irtifakı",
    "Land Title": "Arsa tapusu", "Shared Title": "Hisseli tapu", "No Title Deed": "Tapu yok",
    "Cooperative Share": "Kooperatif hissesi", "Freehold Title": "Müstakil tapu (Freehold Title)",
    "Foreign Owner": "Yabancı uyruklu malik (Foreign Owner)", "Usufruct Right": "İntifa hakkı",
    # ısıtma
    "Combi Boiler": "Kombi", "Central": "Merkezi", "Central (Metered)": "Merkezi (pay ölçer)",
    "Underfloor Heating": "Yerden ısıtma", "Air Conditioning": "Klima", "Stove": "Soba",
    "Gas Stove": "Doğalgaz sobası", "No Heating": "Isıtma yok", "Fan Coil Unit": "Fancoil",
    "Floor Radiator": "Kat kaloriferi",
    # yakıt
    "Natural Gas": "Doğalgaz", "Electricity": "Elektrik",
    # bina durumu / yapı tipi
    "Second-hand": "İkinci el", "New": "Sıfır", "Under Construction": "Yapım aşamasında",
    "Reinforced Concrete": "Betonarme", "Masonry": "Yığma", "Steel": "Çelik",
    # kullanım / eşya / kredi / takas
    "Vacant": "Boş", "Tenant-occupied": "Kiracılı", "Owner-occupied": "Mülk sahibi oturuyor",
    "Furnished": "Eşyalı", "Unfurnished": "Eşyasız",
    "Eligible": "Krediye uygun", "Not Eligible": "Krediye uygun değil",
    "Yes": "Evet", "No": "Hayır",
    # kat türü
    "21 and Above": "21. kat ve üzeri", "Basement": "Bodrum kat", "Basement & Ground": "Bodrum ve zemin",
    "Entrance Floor": "Giriş katı", "Garden Floor": "Bahçe katı", "Ground Floor": "Zemin kat",
    "Mid Floor": "Ara kat", "Penthouse": "Çatı katı (Penthouse)", "Raised Ground": "Yüksek giriş",
    "Semi-Basement": "Yarı bodrum", "Sub-level 1": "Kot 1", "Sub-level 2": "Kot 2", "Sub-level 3": "Kot 3",
    "Terrace Floor": "Teras katı", "Top Floor": "En üst kat", "Villa Floor": "Villa katı",
}
CEPHE = {"Kuzey": "North", "Güney": "South", "Doğu": "East", "Batı": "West"}

OZELLIK_ADI = {
    "net_sqm": "Net m²", "gross_sqm": "Brüt m²", "rooms": "Oda sayısı", "halls": "Salon sayısı",
    "bathroom_count": "Banyo sayısı", "net_brut_orani": "Net/brüt oranı", "oda_basina_m2": "Oda başına m²",
    "floor": "Kat", "total_floors": "Binadaki kat sayısı", "kat_orani": "Kat / toplam kat",
    "en_ust_kat": "En üst kat mı", "yer_alti": "Yer altında mı", "giris_kati": "Giriş katı mı",
    "building_age": "Bina yaşı", "is_in_complex": "Site içinde mi", "maintenance_fee": "Aidat (TL)",
    "aidat_m2": "m² başına aidat", "cephe_kuzey": "Kuzey cephe", "cephe_guney": "Güney cephe",
    "cephe_dogu": "Doğu cephe", "cephe_bati": "Batı cephe", "cephe_sayisi": "Cephe sayısı",
    "son_guncelleme_gun": "İlan güncelliği (gün)", "supheli_m2": "Şüpheli m² işareti",
    "durum_celiskisi": "Durum çelişkisi işareti", "district": "İlçe", "floor_category": "Kat türü",
    "building_condition": "Bina durumu", "building_type": "Yapı tipi", "deed_status": "Tapu durumu",
    "credit_eligible": "Krediye uygunluk", "usage_status": "Kullanım durumu", "furnished": "Eşya durumu",
    "heating_type": "Isıtma", "fuel_type": "Yakıt", "exchange": "Takas", "mahalle": "Mahalle",
}
EVET_HAYIR = {"en_ust_kat", "yer_alti", "giris_kati", "is_in_complex", "cephe_kuzey", "cephe_guney",
              "cephe_dogu", "cephe_bati", "supheli_m2", "durum_celiskisi"}


def tr(deger):
    return ETIKET.get(deger, deger)


def milyon(tl):
    return f"{tl / 1e6:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " milyon TL"


def binlik(sayi):
    return f"{sayi:,.0f}".replace(",", ".")


st.set_page_config(page_title="İstanbul Ev Fiyat Tahmini", page_icon="🏠", layout="wide")

# ================================================================ Model ve veri (bir kez yüklenir)
@st.cache_resource(show_spinner="Model yükleniyor...")
def model_yukle():
    from src import tahmin  # model, kategori listeleri ve özellik üretimi burada
    return tahmin


@st.cache_data(show_spinner="Veri yükleniyor...")
def veri_yukle():
    return pd.read_csv(TEMIZ_VERI, encoding="utf-8-sig")


tahmin = model_yukle()
veri = veri_yukle()

st.title("🏠 İstanbul Daire Fiyat Tahmini")

# ================================================================ Sol panel: ev bilgileri
kenar = st.sidebar
kenar.header("Ev bilgileri")


def kategori_sec(yer, etiket, alan, varsayilan=BILMIYORUM):
    """KATEGORI_LISTESI'nden açılır menü. 'Bilmiyorum' -> None (model eksik değerle çalışır)."""
    secenekler = [BILMIYORUM] + [v for v in tahmin.KATEGORI_LISTESI[alan] if v != "Bilinmiyor"]
    secim = yer.selectbox(etiket, secenekler, index=secenekler.index(varsayilan), format_func=tr, key=alan)
    return None if secim == BILMIYORUM else secim


# --- Konum
ilanlar_ilce = veri["district"].value_counts()
ilceler = sorted(ilanlar_ilce.index, key=lambda s: s.replace("İ", "I").replace("Ş", "S").replace("Ç", "C")
                 .replace("Ü", "U").replace("Ö", "O"))
ilce = kenar.selectbox("İlçe", ilceler, index=ilceler.index("Kadıköy"),
                       format_func=lambda i: f"{i} ({ilanlar_ilce[i]} ilan)")
mah_sayim = veri.loc[veri["district"] == ilce, "neighborhood"].value_counts()
mahalleler = sorted(mah_sayim.index)
mahalle = kenar.selectbox("Mahalle", mahalleler,
                          index=mahalleler.index("Caddebostan") if "Caddebostan" in mahalleler else 0,
                          format_func=lambda m: f"{m} ({mah_sayim[m]} ilan)")
if mah_sayim[mahalle] < AZ_VERI:
    kenar.caption("ℹ️ Az veri, ilçe ortalamasına yakın tahmin edilir.")

# --- Büyüklük
kenar.subheader("Büyüklük")
s1, s2 = kenar.columns(2)
net = s1.number_input("Net m²", min_value=10, max_value=2000, value=120, step=5)
brut = s2.number_input("Brüt m² (opsiyonel)", min_value=10, max_value=3000, value=None, step=5,
                       placeholder=BILMIYORUM)
s1, s2, s3 = kenar.columns(3)
oda = s1.number_input("Oda", min_value=1, max_value=14, value=3)
salon = s2.number_input("Salon", min_value=0, max_value=5, value=1)
banyo = s3.selectbox("Banyo", [BILMIYORUM] + list(range(1, 11)), index=1)

# --- Kat
kenar.subheader("Kat")
s1, s2 = kenar.columns(2)
kat = s1.selectbox("Kat", [BILMIYORUM] + list(range(-3, 56)), index=6,
                   format_func=lambda k: {0: "0 (giriş)", -1: "-1 (bodrum)"}.get(k, str(k)))
toplam_kat = s2.selectbox("Binadaki kat", [BILMIYORUM] + list(range(1, 56)), index=5)
kat_turu = kategori_sec(kenar, "Kat türü (opsiyonel)", "floor_category")

# --- Bina
kenar.subheader("Bina")
yas = kenar.slider("Bina yaşı", 0, 100, 25)
bina_durumu = kategori_sec(kenar, "Bina durumu", "building_condition")
tapu = kategori_sec(kenar, "Tapu durumu", "deed_status", "Condominium Title")
kredi = kategori_sec(kenar, "Krediye uygunluk", "credit_eligible")
isitma = kategori_sec(kenar, "Isıtma", "heating_type")
yakit = kategori_sec(kenar, "Yakıt", "fuel_type")
kullanim = kategori_sec(kenar, "Kullanım durumu", "usage_status")
esya = kategori_sec(kenar, "Eşya", "furnished")
takas = kategori_sec(kenar, "Takas", "exchange")
site = kenar.checkbox("Site içinde")
aidat = kenar.number_input("Aylık aidat, TL (opsiyonel)", min_value=0, max_value=200_000, value=None,
                           step=100, placeholder=BILMIYORUM)
cepheler = kenar.multiselect("Cephe (boş = bilmiyorum)", list(CEPHE))

# ================================================================ Ham ilan (temiz_veri.csv sütun adları)
bos = lambda d: None if d == BILMIYORUM else d
ev = {
    "district": ilce, "neighborhood": mahalle, "net_sqm": float(net),
    "gross_sqm": None if brut is None else float(brut), "rooms": float(oda), "halls": float(salon),
    "bathroom_count": bos(banyo), "floor": bos(kat), "total_floors": bos(toplam_kat), "floor_category": kat_turu,
    "building_age": float(yas), "building_condition": bina_durumu, "building_type": None, "deed_status": tapu,
    "credit_eligible": kredi, "heating_type": isitma, "fuel_type": yakit, "usage_status": kullanim,
    "furnished": esya, "exchange": takas, "is_in_complex": int(site),
    "maintenance_fee": None if aidat is None else float(aidat),
    # veri setindeki yazımla aynı sıra: "North, South, East, West"
    "orientation": ", ".join(v for v in CEPHE.values() if v in {CEPHE[c] for c in cepheler}) or None,
}

# Mantık kontrolleri
mantik = []
if brut is not None and net > brut:
    mantik.append(f"Net m² ({net}) brüt m²'den ({brut}) büyük olamaz.")
if kat != BILMIYORUM and toplam_kat != BILMIYORUM and kat > toplam_kat:
    mantik.append(f"Kat ({kat}) binadaki toplam kattan ({toplam_kat}) büyük olamaz.")
for m in mantik:
    kenar.warning(m)


def tahmin_al(evler):
    """Ham ev listesi -> (tahmin TL dizisi, SHAP katkıları, X, uyarılar)"""
    X, uyarilar = tahmin.ozellik_uret(pd.DataFrame(evler))
    fiyat, katki = tahmin.tahmin_ve_aciklama(X)
    return fiyat, katki, X, uyarilar


fiyatlar, katkilar, X, uyarilar = tahmin_al([ev])
fiyat, katki = float(fiyatlar[0]), katkilar[0]
m2 = fiyat / net

# ================================================================ Sağ panel: sonuç
for m in mantik:
    st.warning("⚠️ " + m)
if uyarilar:
    st.warning("**Modelin tanımadığı / tamamladığı girdiler:**\n\n" + "\n".join(f"- {u}" for u in uyarilar))

# --- 1. Tahmin
st.subheader(f"{ilce} / {mahalle} · {net} m² · {oda}+{salon} · {yas} yaşında")
s1, s2, s3 = st.columns([1.4, 1, 1.4])
s1.metric("Tahmini fiyat", milyon(fiyat))
s2.metric("m² fiyatı", f"{binlik(m2)} TL")
alt, ust_sinir = milyon(fiyat * (1 - MEDYAN_HATA)).split()[0], milyon(fiyat * (1 + MEDYAN_HATA))
s3.metric("Tipik aralık (±%13)", f"{alt} – {ust_sinir}")
st.caption("±%13 test setindeki medyan hatadır; ilanların yaklaşık yarısında gerçek fiyat bu aralığın "
           "dışındadır. Model Mart 2026 ilan fiyatlarıyla eğitildi.")

# --- 2. Bağlam
mah_ilan = veri[(veri["district"] == ilce) & (veri["neighborhood"] == mahalle)]
ilce_ilan = veri[veri["district"] == ilce]
s1, s2 = st.columns(2)
for kutu, ad, grup in [(s1, f"{mahalle} medyan m² fiyatı", mah_ilan), (s2, f"{ilce} medyan m² fiyatı", ilce_ilan)]:
    medyan = grup["price_per_sqm"].median()
    kutu.metric(f"{ad} ({len(grup)} ilan)", f"{binlik(medyan)} TL",
                f"Bu ev {abs(m2 / medyan - 1) * 100:.0f}% {'yukarıda' if m2 >= medyan else 'aşağıda'}",
                delta_color="off")

# --- 3. Neden bu fiyat?
st.markdown("### Neden bu fiyat?")


def deger_yazisi(ozellik, deger):
    if pd.isna(deger):
        return "bilinmiyor"
    if ozellik == "mahalle":
        return str(deger).split(" / ", 1)[1]
    if ozellik in EVET_HAYIR:
        return "evet" if deger == 1 else "hayır"
    if isinstance(deger, str):
        return tr(deger)
    return f"{deger:.2f}".rstrip("0").rstrip(".") if deger % 1 else binlik(deger)


baslangic, parcalar = katki[-1], pd.Series(katki[:-1], index=X.columns)
enler = parcalar.abs().sort_values(ascending=False).index[:10]
adimlar = [(f"{OZELLIK_ADI[o]} = {deger_yazisi(o, X.iloc[0][o])}", parcalar[o]) for o in enler]
adimlar.append(("Diğer özellikler", parcalar.drop(enler).sum()))
etiketler, degisim, yazilar, log_toplam = ["İstanbul ortalaması"], [np.exp(baslangic)], [milyon(np.exp(baslangic))], baslangic
for ad, k in adimlar:
    # Log ölçeğinde toplanabilir katkıyı TL adımına çevir; yüzde = exp(katkı) - 1
    etiketler.append(ad)
    degisim.append(np.exp(log_toplam + k) - np.exp(log_toplam))
    yazilar.append(f"{(np.exp(k) - 1) * 100:+.0f}%")
    log_toplam += k
etiketler.append("Tahmin")
degisim.append(fiyat)
yazilar.append(milyon(fiyat))
sekil = go.Figure(go.Waterfall(
    orientation="h", measure=["absolute"] + ["relative"] * len(adimlar) + ["total"],
    y=etiketler, x=degisim, text=yazilar, textposition="outside",
    increasing={"marker": {"color": "#2e9e5b"}}, decreasing={"marker": {"color": "#d64545"}},
    totals={"marker": {"color": "#3b6fd8"}}, connector={"line": {"color": "#999", "width": 1}},
))
sekil.update_layout(height=520, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="TL",
                    yaxis=dict(autorange="reversed"), showlegend=False)
st.plotly_chart(sekil, width="stretch")
st.caption("Her çubuk, o özelliğin fiyatı İstanbul ortalamasından yüzde kaç yukarı/aşağı ittiğini gösterir "
           "(SHAP katkıları). Katkılar çarpımsaldır: sıraları değişse de sonuç aynı tahmine ulaşır.")

# --- 4. Ya şöyle olsaydı?
st.markdown("### Ya şöyle olsaydı?")
st.caption("Diğer her şey sabitken tek bir özellik değişirse tahmin nasıl değişir?")


def senaryo_grafigi(alan, degerler, simdiki, x_adi):
    fiyat_serisi, _, _, _ = tahmin_al([{**ev, alan: float(d)} for d in degerler])
    sekil = go.Figure(go.Scatter(x=list(degerler), y=fiyat_serisi / 1e6, mode="lines+markers",
                                 line=dict(color="#3b6fd8"), marker=dict(size=4),
                                 hovertemplate=x_adi + ": %{x}<br>%{y:.2f} milyon TL<extra></extra>"))
    if simdiki in degerler:
        sekil.add_trace(go.Scatter(x=[simdiki], y=[fiyat / 1e6], mode="markers", marker=dict(size=14, color="#d64545"),
                                   hovertemplate="Şu anki: %{y:.2f} milyon TL<extra></extra>"))
    sekil.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                        xaxis_title=x_adi, yaxis_title="Tahmin (milyon TL)")
    st.plotly_chart(sekil, width="stretch")


sekme_yas, sekme_kat = st.tabs(["Bina yaşı", "Kat"])
with sekme_yas:
    senaryo_grafigi("building_age", list(range(0, 61)), yas, "Bina yaşı")
with sekme_kat:
    ust = int(toplam_kat) if toplam_kat != BILMIYORUM else 20
    senaryo_grafigi("floor", list(range(-1, ust + 1)), kat, "Kat")
    if toplam_kat == BILMIYORUM:
        st.caption("Binadaki kat sayısı bilinmediği için 20. kata kadar gösteriliyor.")
    if kat_turu:
        st.caption(f"Kat türü '{tr(kat_turu)}' sabit tutuldu; gerçekte kat değişince kat türü de değişebilir.")

# --- 5. Benzer ilanlar
st.markdown("### Benzer ilanlar")


def benzerler(grup):
    g = grup[grup["net_sqm"].between(net * 0.8, net * 1.2)].copy()
    g["_uzaklik"] = (g["net_sqm"] - net).abs() / net + (g["building_age"] - yas).abs().fillna(50) / 50
    return g.sort_values("_uzaklik").head(5)


benzer, kaynak = benzerler(mah_ilan), mahalle
if benzer.empty:
    benzer, kaynak = benzerler(ilce_ilan), f"{ilce} (mahallede benzer ilan yok)"
if benzer.empty:
    st.info("Bu büyüklükte benzer ilan bulunamadı.")
else:
    st.caption(f"Kaynak: {kaynak} · net m² ±%20 · büyüklük ve yaşça en yakın 5 ilan")
    tablo = pd.DataFrame({
        "Mahalle": benzer["neighborhood"], "Net m²": benzer["net_sqm"].astype(int),
        "Oda": benzer["rooms"].fillna(0).astype(int).astype(str) + "+" + benzer["halls"].fillna(0).astype(int).astype(str),
        "Yaş": benzer["building_age"].astype("Int64"), "Kat": benzer["floor"].astype("Int64"),
        "İlan fiyatı": benzer["price"].map(milyon), "m² fiyatı (TL)": benzer["price_per_sqm"].map(binlik),
    })
    st.dataframe(tablo, hide_index=True, width="stretch")

# --- 6. Karşılaştırma
st.markdown("### Karşılaştırma")
if "karsilastirma" not in st.session_state:
    st.session_state.karsilastirma = []
liste = st.session_state.karsilastirma
s1, s2, _ = st.columns([1, 1, 3])
if s1.button("➕ Karşılaştırmaya ekle", disabled=len(liste) >= 5):
    liste.append({
        "Konum": f"{ilce} / {mahalle}", "Net m²": net, "Oda": f"{oda}+{salon}", "Yaş": yas,
        "Kat": "?" if kat == BILMIYORUM else kat, "Isıtma": tr(isitma) if isitma else "?",
        "Site": "evet" if site else "hayır", "Tahmin": milyon(fiyat), "m² fiyatı (TL)": binlik(m2),
    })
if s2.button("🗑️ Temizle", disabled=not liste):
    liste.clear()
if len(liste) >= 5:
    st.caption("En fazla 5 ev karşılaştırılabilir.")
if liste:
    # Evler sütun olarak yan yana
    st.dataframe(pd.DataFrame(liste, index=[f"Ev {i + 1}" for i in range(len(liste))]).T.astype(str),
                 width="stretch")
else:
    st.caption("Henüz ev eklenmedi. Soldaki bilgileri değiştirip farklı evleri ekleyebilirsiniz.")
