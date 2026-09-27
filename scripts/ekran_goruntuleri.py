"""
README EKRAN GÖRÜNTÜLERİ
Kullanım:  ./venv/bin/python scripts/ekran_goruntuleri.py
    (bir kez: ./venv/bin/python -m pip install playwright && ./venv/bin/python -m playwright install chromium)

app.py ve izle.py'yi arka planda başlatır, Playwright ile açar ve docs/img/ klasörüne
ekran görüntülerini kaydeder. Sonunda iki uygulamayı da kapatır.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

KOK = Path(__file__).resolve().parent.parent
CIKTI = KOK / "docs" / "img"
CIKTI.mkdir(parents=True, exist_ok=True)
STREAMLIT = str(Path(sys.executable).parent / "streamlit")


def baslat(dosya, port):
    """Streamlit uygulamasını açık temayla başlatır ve hazır olmasını bekler."""
    surec = subprocess.Popen([STREAMLIT, "run", dosya, "--server.port", str(port), "--server.headless", "true",
                              "--theme.base", "light", "--browser.gatherUsageStats", "false",
                              "--client.toolbarMode", "viewer"],
                             cwd=KOK, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
            return surec
        except OSError:
            time.sleep(1)
    surec.kill()
    sys.exit(f"{dosya} başlatılamadı")


def bekle(sayfa, metin, grafik=0):
    """Sayfada metin ve en az `grafik` kadar plotly grafiği çizilene kadar bekler."""
    sayfa.get_by_text(metin).first.wait_for(timeout=120_000)
    if grafik:
        sayfa.wait_for_function(f"document.querySelectorAll('.js-plotly-plot .main-svg').length >= {grafik}",
                                timeout=60_000)
    sayfa.wait_for_timeout(1500)  # animasyonlar otursun


def kaydet(hedef, ad):
    hedef.screenshot(path=CIKTI / ad)
    print(f"  docs/img/{ad}")


def app_goruntuleri(tarayici):
    # Streamlit ana alanı kendi içinde kaydırılır; tüm sayfayı almak için pencere uzun tutulur
    sayfa = tarayici.new_page(viewport={"width": 1500, "height": 4300}, device_scale_factor=1)
    sayfa.goto("http://localhost:8601")
    bekle(sayfa, "Tahmini fiyat", grafik=2)

    # Karşılaştırma tablosu dolu görünsün: 25 yaşındaki evi ekle, yaşı 5'e çekip tekrar ekle
    sayfa.get_by_role("button", name="Karşılaştırmaya ekle").click()
    sayfa.wait_for_timeout(1500)
    kaydirici = sayfa.get_by_role("slider").first
    kaydirici.focus()
    for _ in range(20):
        kaydirici.press("ArrowLeft")
    bekle(sayfa, "5 yaşında", grafik=2)
    sayfa.get_by_role("button", name="Karşılaştırmaya ekle").click()
    sayfa.wait_for_timeout(1500)
    for _ in range(20):
        kaydirici.press("ArrowRight")
    bekle(sayfa, "25 yaşında", grafik=2)

    kaydet(sayfa, "app_tam_sayfa.png")
    sayfa.set_viewport_size({"width": 1500, "height": 900})
    sayfa.wait_for_timeout(1500)
    kaydet(sayfa, "app_ust.png")
    sayfa.set_viewport_size({"width": 1500, "height": 4300})
    sayfa.wait_for_timeout(1500)
    grafikler = sayfa.locator('[data-testid="stPlotlyChart"]')
    kaydet(grafikler.nth(0), "app_selale.png")
    kaydet(sayfa.locator('[data-testid="stTabs"]').first, "app_senaryo.png")
    kaydet(sayfa.locator('[data-testid="stDataFrame"]').last, "app_karsilastirma.png")
    sayfa.close()


def izle_goruntuleri(tarayici):
    sayfa = tarayici.new_page(viewport={"width": 1500, "height": 1720}, device_scale_factor=1)
    sayfa.goto("http://localhost:8602")
    bekle(sayfa, "Uyuşmayan ağaç")
    kaydet(sayfa, "izle.png")
    sayfa.close()


if __name__ == "__main__":
    surecler = [baslat("app.py", 8601), baslat("izle.py", 8602)]
    try:
        with sync_playwright() as p:
            tarayici = p.chromium.launch()
            print("Ekran görüntüleri:")
            app_goruntuleri(tarayici)
            izle_goruntuleri(tarayici)
            tarayici.close()
    finally:
        for s in surecler:
            s.terminate()
