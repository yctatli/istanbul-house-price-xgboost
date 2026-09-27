"""
Projedeki tüm klasör ve dosya yolları. Bir dosyanın yeri değişirse sadece burası güncellenir.
"""
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

# Veri
HAM_KLASOR = KOK / "data" / "raw"                  # dokunulmamış ham veri
TEMIZ_KLASOR = KOK / "data" / "processed"          # clear_data.py çıktıları
OZELLIK_KLASOR = KOK / "data" / "features"         # ozellik.py çıktıları
TEMIZ_VERI = TEMIZ_KLASOR / "temiz_veri.csv"
OZELLIKLER = OZELLIK_KLASOR / "ozellikler.csv"
OZELLIK_LISTESI = OZELLIK_KLASOR / "ozellik_listesi.json"

# Modeller
MODEL_KLASOR = KOK / "models"
FINAL_MODEL = MODEL_KLASOR / "xgb_final.json"
FINAL_AYARLAR = MODEL_KLASOR / "final_ayarlar.json"
BASELINE_MODEL = MODEL_KLASOR / "baseline" / "xgb_model.json"   # 4. adım modeli
AYRIM = MODEL_KLASOR / "baseline" / "ayrim.json"                 # eğitim/doğrulama/test ayrımı

# Çıktılar
GRAFIK_KLASOR = KOK / "plots"
RAPOR_KLASOR = KOK / "reports"
