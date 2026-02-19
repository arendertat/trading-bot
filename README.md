# Binance USDT-M Futures Trading Bot

Paper trading ve canlı trading için üretim kalitesinde otomatik kripto futures botu.

> ⚠️ **Risk Uyarısı**: Bu bot kaldıraçlı futures işlemleri yapar. Kaybetmeyi göze alabileceğin sermayeyle kullan. Canlıya geçmeden önce en az 7 gün paper modda test et.

---

## Hızlı Başlangıç

### 1. Gereksinimleri Yükle

```bash
pip install -r requirements.txt
```

### 2. .env Dosyasını Hazırla

```bash
cp .env.example .env
```

`.env` dosyasını aç ve doldur:

```env
BINANCE_API_KEY=buraya_api_key_yaz
BINANCE_API_SECRET=buraya_api_secret_yaz
```

> Testnet API key almak için: https://testnet.binancefuture.com → Kayıt ol → API Management

### 3. Config Dosyasını Hazırla

```bash
cp config/config.example.json config/config.json
```

En az şunları kontrol et:
- `exchange.testnet`: `true` bırak (canlı için `false`)
- `risk.risk_per_trade_pct`: `0.01` = bakiyenin %1'i (güvenli başlangıç)

### 4. Config'i Doğrula

```bash
python -m bot.config.loader config/config.json
```

Başarılı çıktı:
```
[OK] Config validation successful
  Mode: PAPER_LIVE
  Exchange: binance
  Max positions: 2
  Risk per trade: 1.00%
```

### 5. Botu Çalıştır

```bash
python -m bot.main
```

Ctrl+C ile güvenli durdurma. Kapatınca tüm pozisyonları kapat:

```bash
python -m bot.main --close-all
```

---

## Çalışma Modu

### PAPER_LIVE (Önerilen Başlangıç)

Gerçek exchange'e bağlanır, piyasa verisi alır ama **asla gerçek emir göndermez**. Kâr/zarar simüle edilir.

```json
"mode": "PAPER_LIVE"
```

### LIVE (Gerçek Para)

Gerçek emirler gönderir. Bunu kullanmak için:

1. Config'de `"testnet": false` yap
2. `.env` dosyasına ekle: `LIVE_TRADING_CONFIRMED=true`
3. Gerçek Binance API key'i kullan

---

## Config Dosyası Açıklaması

`config/config.json` içindeki tüm alanlar ve ne işe yaradıkları:

### `exchange` — Borsa Ayarları

```json
"exchange": {
  "testnet": true,          // true = Binance testnet, false = gerçek borsa
  "margin_mode": "ISOLATED" // Her pozisyon ayrı margin (ISOLATED önerilir)
}
```

### `universe` — Hangi Sembolleri İzle

```json
"universe": {
  "min_24h_volume_usdt": 100000000,  // Min günlük hacim: 100M$
  "max_spread_pct": 0.0005,          // Max spread: %0.05
  "max_abs_funding_rate": 0.0015,    // Max funding rate: %0.15
  "max_monitored_symbols": 6,        // En fazla kaç sembol izlensin
  "whitelist": ["BTCUSDT", "ETHUSDT"], // Sadece bunları izle (boş = otomatik seç)
  "blacklist": ["XYZUSDT"]           // Bunları hiç izleme
}
```

> `whitelist` boş bırakılırsa bot hacim/spread kriterlerine göre en iyi 6 sembolü otomatik seçer.

### `risk` — Risk Yönetimi

```json
"risk": {
  "risk_per_trade_pct": 0.01,         // İşlem başına risk: %1 (bakiyenin)
  "max_total_open_risk_pct": 0.025,   // Toplam açık risk: %2.5
  "max_open_positions": 2,            // Aynı anda max açık pozisyon
  "max_same_direction_positions": 2,  // Aynı yönde max pozisyon (LONG veya SHORT)
  "daily_stop_pct": -0.04,            // Günlük zarar limiti: -%4 → bot durur
  "weekly_stop_pct": -0.1,            // Haftalık zarar limiti: -%10 → bot durur
  "pause_days_after_weekly_stop": 7,  // Haftalık stop sonrası kaç gün beklesin
  "reduced_risk_after_pause_pct": 0.005, // Duraklamadan sonra düşürülmüş risk: %0.5
  "reduced_risk_days": 3,             // Kaç gün düşük riskle çalışsın
  "high_vol_risk_reduction_pct": 0.005, // Yüksek volatilite rejiminde risk: %0.5
  "cooldown_after_sl_bars": 0,        // SL sonrası aynı sembole giriş yasağı (bar sayısı, 0=kapalı)
  "max_net_exposure_pct": 2.0,        // Net yön maruziyeti limiti (2.0 = kapalı)
  "max_single_symbol_exposure_pct": 1.0 // Tek sembol limiti (1.0 = kapalı)
}
```

> Yüzdeler ondalık: `0.01` = %1, `0.04` = %4

### `regime` — Piyasa Rejimi Tespiti

```json
"regime": {
  "trend_adx_min": 25,         // ADX bu değerin üstündeyse TREND rejimi
  "range_adx_max": 20,         // ADX bu değerin altındaysa RANGE rejimi
  "high_vol_atr_z": 1.5,       // ATR z-score bu değerin üstündeyse HIGH_VOL
  "confidence_threshold": 0.55, // Rejim güven eşiği (0-1)
  "adaptive_regime": false,    // true = ADX eşiklerini otomatik ayarla
  "adaptive_adx_window": 8640  // Adaptive için kaç bar geçmişe bak (8640 = ~30 gün)
}
```

### `strategies` — Strateji Ayarları

#### Trend Pullback (Trend Geri Çekilmesi)

```json
"trend_pullback": {
  "enabled": true,
  "stop_pct": 0.01,              // Stop loss: %1
  "target_r_multiple": 1.5,      // Hedef: 1.5R (riskin 1.5 katı kâr)
  "pullback_rsi_long_min": 40,   // LONG için RSI alt sınırı
  "pullback_rsi_long_max": 50,   // LONG için RSI üst sınırı
  "pullback_rsi_short_min": 50,  // SHORT için RSI alt sınırı
  "pullback_rsi_short_max": 60,  // SHORT için RSI üst sınırı
  "ema20_band_pct": 0.002,       // Fiyat EMA20'ye bu kadar yakın olmalı (%0.2)
  "trail_after_r": 1.0,          // 1R kârdan sonra trailing stop aktif
  "atr_trail_mult": 2.0,         // Trailing stop mesafesi: 2 * ATR
  "dynamic_stop_enabled": false, // true = sabit % yerine ATR-bazlı stop
  "stop_atr_multiplier": 1.5,    // ATR-bazlı stop: 1.5 * ATR mesafe
  "use_book_imbalance": false,   // true = order book bid/ask oranı filtresi
  "book_imbalance_threshold": 1.2 // LONG için bid/ask > 1.2 olmalı
}
```

#### Trend Breakout (Kırılım)

```json
"trend_breakout": {
  "enabled": true,
  "stop_pct": 0.01,
  "breakout_lookback_bars": 20,    // 20 barlık yüksek/düşük kır
  "breakout_volume_z_min": 1.0,    // Kırılım için minimum hacim z-score
  "atr_trail_mult": 2.5,
  "dynamic_stop_enabled": false,
  "stop_atr_multiplier": 1.5,
  "use_book_imbalance": false,
  "book_imbalance_threshold": 1.2
}
```

#### Range Mean Reversion (Yatay Piyasa)

```json
"range_mean_reversion": {
  "enabled": true,
  "stop_pct": 0.008,           // %0.8 stop
  "target_r_multiple": 1.2,    // Hedef 1.2R
  "rsi_long_extreme": 25,      // RSI < 25 → aşırı satım → LONG al
  "rsi_short_extreme": 75,     // RSI > 75 → aşırı alım → SHORT al
  "dynamic_stop_enabled": false,
  "stop_atr_multiplier": 1.5
}
```

### `leverage` — Kaldıraç

```json
"leverage": {
  "trend": 2.0,    // Trend rejiminde max 2x kaldıraç
  "range": 1.5,    // Range rejiminde max 1.5x
  "high_vol": 1.0  // Yüksek volatilite → kaldıraç yok
}
```

### `execution` — Emir Ayarları

```json
"execution": {
  "paper_slippage_limit_pct": 0.0002,  // Simüle slippage: %0.02 (limit emir)
  "paper_slippage_stop_pct": 0.001,    // Simüle slippage: %0.1 (stop emir)
  "maker_fee_pct": 0.0002,             // Maker komisyon: %0.02
  "taker_fee_pct": 0.0004,             // Taker komisyon: %0.04
  "enable_funding_in_paper": false     // true = funding maliyetini simüle et
}
```

### `dashboard` — Web Arayüzü

```json
"dashboard": {
  "enabled": false,        // true = web dashboard aktif
  "host": "127.0.0.1",    // Sadece lokal erişim (dışarıdan: "0.0.0.0")
  "port": 8080             // http://localhost:8080
}
```

Dashboard açıkken şu bilgileri gösterir:
- Equity eğrisi (Chart.js)
- Açık pozisyonlar ve PnL
- Kapalı işlemler geçmişi
- Strateji performansı

### `notifications` — Telegram Bildirimleri

```json
"notifications": {
  "telegram_enabled": false,
  "telegram_token_env": "TELEGRAM_BOT_TOKEN",   // .env'den okur
  "telegram_chat_id_env": "TELEGRAM_CHAT_ID",   // .env'den okur
  "daily_report_time_utc": "00:05"              // Günlük rapor saati
}
```

---

## Opsiyonel Özellikler

### 4h Çoklu Zaman Dilimi Onayı

Trend stratejilerinde 4 saatlik EMA yapısını da kontrol eder:

```json
"trend_pullback": {
  "use_4h_confirmation": true
}
```

### Adaptive ADX Eşikleri

Piyasa koşullarına göre TREND/RANGE eşiklerini otomatik ayarlar:

```json
"regime": {
  "adaptive_regime": true,
  "adaptive_adx_window": 8640
}
```

### SL Cooldown

Stop-loss'tan sonra aynı sembole belirli süre girme:

```json
"risk": {
  "cooldown_after_sl_bars": 6
}
```

### Order Book Filtresi

Giriş sinyalini bid/ask hacim oranıyla doğrula:

```json
"trend_pullback": {
  "use_book_imbalance": true,
  "book_imbalance_threshold": 1.2
}
```

### Funding Rate Simülasyonu

Paper modda her 8 saatte bir funding maliyetini uygula:

```json
"execution": {
  "enable_funding_in_paper": true
}
```

---

## Sunucudan Dashboard'a Erişim

### Seçenek 1 — SSH Tunnel (En Güvenli)

Kendi bilgisayarında çalıştır:
```bash
ssh -L 8080:127.0.0.1:8080 kullanici@sunucu_ip
```
Sonra: http://localhost:8080

### Seçenek 2 — Direkt Erişim

```json
"dashboard": {
  "host": "0.0.0.0",
  "port": 8080
}
```

Firewall aç:
```bash
sudo ufw allow 8080/tcp
# veya sadece belirli IP'den:
sudo ufw allow from KENDİ_IP to any port 8080
```

Sonra: http://SUNUCU_IP:8080

---

## Testleri Çalıştır

```bash
# Tüm testler
python -m pytest tests/ -q --ignore=tests/test_exchange_layer.py

# Detaylı çıktı
python -m pytest tests/ -v --ignore=tests/test_exchange_layer.py

# Belirli modül
python -m pytest tests/test_risk_engine.py -v
```

---

## Proje Yapısı

```
trading-bot/
├── bot/
│   ├── config/         # Config yükleme ve doğrulama
│   ├── core/           # Sabitler ve temel tipler
│   ├── data/           # Mum verisi, feature engine, WebSocket
│   ├── exchange/       # Binance API istemcisi
│   ├── execution/      # Pozisyon, trailing stop
│   ├── health/         # Safe mode
│   ├── regime/         # Piyasa rejimi tespiti
│   ├── risk/           # Risk engine, kill switch, korelasyon filtresi
│   ├── state/          # Pozisyon state, log okuyucu
│   ├── strategies/     # Trend Pullback, Breakout, Range stratejileri
│   ├── universe/       # Sembol seçimi
│   ├── dashboard/      # Web arayüzü (FastAPI)
│   └── main.py         # Giriş noktası
├── config/
│   ├── config.example.json   # Örnek config (buradan kopyala)
│   └── config.json           # Senin config'in (.gitignore'da olmalı)
├── tests/              # Test suite (905 test)
├── logs/               # trades.jsonl, events.jsonl
├── .env.example        # Örnek env dosyası
├── .env                # Senin API key'lerin (.gitignore'da olmalı)
└── requirements.txt
```

---

## Log Dosyaları

Bot `./logs/` klasörüne iki JSONL dosyası yazar:

| Dosya | İçerik |
|-------|--------|
| `trades.jsonl` | Açılan/kapanan pozisyonlar, funding, slippage |
| `events.jsonl` | Sistem olayları, uyarılar, exposure snapshot'ları |

---

## Sık Karşılaşılan Hatalar

**`Config file not found`**
→ `cp config/config.example.json config/config.json` yap

**`Environment variable BINANCE_API_KEY not found`**
→ `.env` dosyası yok veya boş. `cp .env.example .env` yap ve key'leri ekle

**`LIVE TRADING IS NOT ENABLED`**
→ Gerçek trading için `.env`'e `LIVE_TRADING_CONFIRMED=true` ekle

**`Exchange ping failed`**
→ API key yanlış veya testnet/mainnet uyumsuzluğu. Config'deki `testnet` değerini kontrol et

**`ModuleNotFoundError`**
→ `pip install -r requirements.txt` çalıştır

**`max_same_direction_positions cannot exceed max_open_positions`**
→ Config'de `max_same_direction_positions` <= `max_open_positions` olmalı

---

## Güvenlik Notları

- `.env` ve `config/config.json` dosyalarını **asla** git'e commit etme
- `LIVE_TRADING_CONFIRMED=true` sadece gerçekten canlı trade yapmak istediğinde ekle
- Testnet'te en az 1 hafta çalıştır, sonra küçük miktarla canlıya geç
- Günlük ve haftalık stop değerlerini muhafazakar tut başlangıçta


python3 -m bot.backtest --symbols BTCUSDT --start 2024-01-01 --end 2024-06-01 --output results/rapor_v2.json
