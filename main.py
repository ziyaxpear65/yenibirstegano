"""
Steganografi + IPFS + MetaMask (imza doğrulamalı) Giriş Sistemi
------------------------------------------------------------------
- Kullanıcı önce MetaMask ile cüzdanını bağlar ve bir nonce mesajını
  imzalar (personal_sign). Sunucu tarafında imza doğrulanır; sadece
  gerçekten o cüzdanın sahibi giriş yapabilir.
- Gizli mesaj, kullanıcının belirlediği bir parola ile AES-256-GCM
  kullanılarak şifrelenir.
- Şifreli veri, resmin piksel kanallarına LSB (en az anlamlı bit)
  yöntemiyle gömülür. Gömme sırası da parolaya bağlı bir sözde-rastgele
  (PRNG) permütasyon ile karıştırılır; yani parolayı bilmeyen biri hem
  veriyi çözemez hem de verinin resimde nerede olduğunu bilemez.
- Sonuç görsel Pinata üzerinden IPFS'e yüklenir, kullanıcıya IPFS hash
  verilir. Çözme tarafında IPFS hash + parola ile orijinal mesaj geri
  elde edilir.

GÜVENLİK / DÜRÜSTLÜK NOTLARI (lütfen okuyun):
- Bu, eğitim/proje amaçlı, "iyi" düzeyde güçlendirilmiş bir LSB
  steganografi sistemidir. Parola tabanlı şifreleme + karıştırılmış
  gömme sırası, saf LSB'ye göre çok daha güvenlidir; ancak istatistiksel
  steganaliz (ör. chi-square testi) resimde "bir şey gizlendiğini"
  yine de tespit edebilir. Askeri/üretim düzeyinde gizlilik gerektiren
  senaryolar için ayrıca uzman danışmanlığı alınmalı.
- LSB veri gömme kayıpsız (lossless) formatlarda çalışır; bu yüzden
  çıktı her zaman PNG olarak üretilir. JPEG gibi kayıplı formatlara
  kaydedilir/yüklenirse gömülü veri bozulur.
- API anahtarlarını / gizli bilgileri asla kod içine yazmayın; ortam
  değişkeni veya st.secrets kullanın (aşağıda öyle yapılmıştır).
"""

from __future__ import annotations

import base64
import hashlib
import os
import random
import sqlite3
import time
import uuid
from contextlib import contextmanager
from io import BytesIO
from typing import Optional

import json as json_module

import html as html_module

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_javascript import st_javascript

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from eth_account import Account
from eth_account.messages import encode_defunct

# ==========================================================================
# Yapılandırma
# ==========================================================================

def _get_secret(name: str) -> str:
    """secrets.toml dosyası hiç yoksa st.secrets erişimi hata fırlatır;
    bu yüzden güvenli şekilde deniyoruz ve yoksa boş string dönüyoruz."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def _get_config_value(name: str) -> str:
    """Ortam değişkeni veya secrets.toml'dan okur ve başta/sonda kalabilecek
    boşluk/satır sonu gibi karakterleri temizler (HTTP header hatalarına
    yol açtığı için önemli)."""
    value = os.environ.get(name) or _get_secret(name)
    return value.strip() if value else ""


PINATA_API_KEY = _get_config_value("PINATA_API_KEY")
PINATA_SECRET_KEY = _get_config_value("PINATA_SECRET_KEY")

PINATA_ENDPOINT = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_GATEWAY_PREFIX = "https://gateway.pinata.cloud/ipfs/"

PBKDF2_ITERATIONS = 200_000
SALT_LEN = 16          # bytes
NONCE_LEN = 12         # bytes (AES-GCM için standart)
HEADER_BITS = 32       # payload uzunluğunu tutan başlık (4 byte -> 32 bit)

# ADMIN_WALLETS: virgülle ayrılmış cüzdan adresleri. BİLEREK sadece ortam
# değişkeni / secrets.toml'dan okunuyor — uygulamanın kendi arayüzünden
# (veritabanı üzerinden) kimseye admin yetkisi verilemez. Bu, bir saldırganın
# veritabanına yazma erişimi elde etse bile admin olamaması anlamına gelir;
# admin listesini değiştirmenin tek yolu sunucu ortamına erişmektir.
_ADMIN_WALLETS_RAW = _get_config_value("ADMIN_WALLETS")
ADMIN_WALLETS = {
    addr.strip().lower() for addr in _ADMIN_WALLETS_RAW.split(",") if addr.strip()
}

DB_PATH = os.environ.get("STEGO_DB_PATH") or "stego_app.db"

# SUPABASE_DB_URL tanımlıysa kalıcı Postgres (Supabase) kullanılır — Streamlit
# Cloud'un geçici dosya sistemi yüzünden veri kaybolmaz. Tanımlı değilse
# yerel geliştirme için SQLite'a otomatik düşülür.
SUPABASE_DB_URL = _get_config_value("SUPABASE_DB_URL")
USE_POSTGRES = bool(SUPABASE_DB_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

st.set_page_config(page_title="Steganografi + IPFS + MetaMask", page_icon="🔐", layout="wide")


# ==========================================================================
# 0) Kullanıcı veritabanı (Postgres/Supabase veya SQLite) — kimlik,
#    yasaklama, işlem günlüğü
# ==========================================================================
# Not: Veritabanında ASLA parola, şifre çözme anahtarı veya mesaj içeriği
# tutulmuyor — sadece cüzdan adresi, zaman damgası ve IPFS hash'i gibi
# zaten herkese açık/kamusal bilgiler. Yani admin panelini görebilen biri
# bile kullanıcıların gizli mesajlarını OKUYAMAZ; yalnızca "kim, ne zaman,
# hangi işlemi yaptı" bilgisini görür.

def _placeholder() -> str:
    return "%s" if USE_POSTGRES else "?"


@contextmanager
def _db():
    if USE_POSTGRES:
        conn = psycopg2.connect(SUPABASE_DB_URL)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _dict_cursor(conn):
    """Postgres'te de SQLite'taki gibi row['sutun'] erişimi için dict-benzeri
    imleç döndürür."""
    if USE_POSTGRES:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


def init_db():
    global USE_POSTGRES
    try:
        with _db() as conn:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        wallet_address TEXT PRIMARY KEY,
                        first_seen DOUBLE PRECISION NOT NULL,
                        last_seen DOUBLE PRECISION NOT NULL,
                        is_banned BOOLEAN NOT NULL DEFAULT FALSE,
                        banned_reason TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS activity_log (
                        id SERIAL PRIMARY KEY,
                        wallet_address TEXT NOT NULL,
                        action TEXT NOT NULL,
                        ipfs_hash TEXT,
                        success BOOLEAN NOT NULL,
                        timestamp DOUBLE PRECISION NOT NULL
                    )
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        wallet_address TEXT PRIMARY KEY,
                        first_seen REAL NOT NULL,
                        last_seen REAL NOT NULL,
                        is_banned INTEGER NOT NULL DEFAULT 0,
                        banned_reason TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS activity_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        wallet_address TEXT NOT NULL,
                        action TEXT NOT NULL,
                        ipfs_hash TEXT,
                        success INTEGER NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
    except Exception as exc:
        if USE_POSTGRES:
            st.warning(
                f"Kalıcı veritabanına (Supabase) bağlanılamadı, geçici SQLite'a "
                f"düşülüyor. Ban listesi/loglar bu oturumda kalıcı olmayacak. Hata: {exc}"
            )
            USE_POSTGRES = False
            init_db()
        else:
            raise


def upsert_user(wallet_address: str):
    wallet_address = wallet_address.lower()
    now = time.time()
    ph = _placeholder()
    banned_default = "FALSE" if USE_POSTGRES else "0"
    with _db() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO users (wallet_address, first_seen, last_seen, is_banned)
            VALUES ({ph}, {ph}, {ph}, {banned_default})
            ON CONFLICT (wallet_address) DO UPDATE SET last_seen = EXCLUDED.last_seen
        """, (wallet_address, now, now))


def is_wallet_banned(wallet_address: str) -> bool:
    wallet_address = wallet_address.lower()
    ph = _placeholder()
    with _db() as conn:
        cur = _dict_cursor(conn)
        cur.execute(f"SELECT is_banned FROM users WHERE wallet_address = {ph}", (wallet_address,))
        row = cur.fetchone()
    return bool(row and row["is_banned"])


def set_wallet_banned(wallet_address: str, banned: bool, reason: str = ""):
    wallet_address = wallet_address.lower()
    ph = _placeholder()
    banned_value = banned if USE_POSTGRES else (1 if banned else 0)
    with _db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE users SET is_banned = {ph}, banned_reason = {ph} WHERE wallet_address = {ph}",
            (banned_value, reason, wallet_address),
        )


def is_admin_wallet(wallet_address: str) -> bool:
    return bool(wallet_address) and wallet_address.lower() in ADMIN_WALLETS


def log_action(wallet_address: str, action: str, ipfs_hash: str = "", success: bool = True):
    ph = _placeholder()
    success_value = success if USE_POSTGRES else (1 if success else 0)
    with _db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO activity_log (wallet_address, action, ipfs_hash, success, timestamp) "
            f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph})",
            (wallet_address.lower(), action, ipfs_hash, success_value, time.time()),
        )


def list_users():
    with _db() as conn:
        cur = _dict_cursor(conn)
        cur.execute("SELECT * FROM users ORDER BY last_seen DESC")
        return cur.fetchall()


def list_recent_activity(limit: int = 100):
    ph = _placeholder()
    with _db() as conn:
        cur = _dict_cursor(conn)
        cur.execute(f"SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT {ph}", (limit,))
        return cur.fetchall()


init_db()


# ==========================================================================
# 1) MetaMask bağlantısı + imza doğrulama
# ==========================================================================

def _new_nonce() -> str:
    return uuid.uuid4().hex


def _login_message(nonce: str) -> str:
    return (
        "Steganografi + IPFS uygulamasına giriş yapıyorsunuz.\n"
        f"Nonce: {nonce}\n"
        "Bu imza yalnızca kimliğinizi doğrulamak için kullanılır, "
        "herhangi bir işlem/transfer başlatmaz."
    )


def verify_signature(message: str, signature: str, expected_address: str) -> bool:
    """personal_sign ile atılan imzayı doğrular, imzalayan adresi karşılaştırır."""
    try:
        encoded = encode_defunct(text=message)
        recovered = Account.recover_message(encoded, signature=signature)
        return recovered.lower() == expected_address.lower()
    except Exception:
        return False


def _build_metamask_js(message: str) -> str:
    """MetaMask'a bağlanıp mesajı imzalatan ve sonucu JSON string olarak
    döndüren JS ifadesini üretir. st_javascript bunu bir async IIFE içine
    sararak çalıştırır, bu yüzden burası tek bir "ifade" (expression)
    olmalı — sayfa yönlendirmesi YAPMAZ, sadece sonucu Python'a döndürür."""
    message_js = json_module.dumps(message)
    return f"""(async () => {{
        try {{
            // Streamlit bu kodu bir iç iframe'de çalıştırır; MetaMask genelde
            // yalnızca gerçek http(s) adresli üst pencereye (window.top)
            // enjekte olur, bu yüzden önce onu deniyoruz.
            const provider = (window.top && window.top.ethereum) ? window.top.ethereum : window.ethereum;
            if (!provider) {{
                return JSON.stringify({{error: "MetaMask bulunamadı. Lütfen tarayıcı eklentisini kurun ve bu site için etkinleştirin."}});
            }}
            const accounts = await provider.request({{method: 'eth_requestAccounts'}});
            const address = accounts[0];
            const signature = await provider.request({{
                method: 'personal_sign',
                params: [{message_js}, address]
            }});
            return JSON.stringify({{address: address, signature: signature}});
        }} catch (err) {{
            return JSON.stringify({{error: (err && err.message) ? err.message : String(err)}});
        }}
    }})()"""


def render_metamask_login():
    """MetaMask bağlantı + imza akışını başlatan buton ve mantığı çizer."""
    nonce = st.session_state["auth_nonce"]
    message = _login_message(nonce)

    if "login_attempt" not in st.session_state:
        st.session_state["login_attempt"] = 0

    if st.button("🦊 MetaMask ile Bağlan & İmzala", type="primary"):
        st.session_state["login_attempt"] += 1

    if st.session_state["login_attempt"] > 0:
        js_code = _build_metamask_js(message)
        # key'i denemeye göre değiştiriyoruz ki her tıklamada bileşen
        # yeniden monte olsun ve MetaMask akışı gerçekten tekrar çalışsın.
        with st.spinner("MetaMask'ta bağlantı ve imza isteğini onaylayın..."):
            result = st_javascript(js_code, key=f"metamask_login_{st.session_state['login_attempt']}")

        if result and result != 0:
            try:
                payload = json_module.loads(result)
            except (TypeError, ValueError):
                payload = {"error": "Beklenmeyen bir yanıt alındı, lütfen tekrar deneyin."}

            if "error" in payload:
                st.error(payload["error"])
            elif payload.get("address") and payload.get("signature"):
                wallet_address = payload["address"]
                if verify_signature(message, payload["signature"], wallet_address):
                    upsert_user(wallet_address)  # ilk/son görülme kaydı
                    if is_wallet_banned(wallet_address):
                        st.error("Bu cüzdan yönetici tarafından engellenmiştir. Erişiminiz yok.")
                        st.session_state["auth_nonce"] = _new_nonce()
                        st.session_state["login_attempt"] = 0
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["wallet"] = wallet_address
                        st.rerun()
                else:
                    st.error("İmza doğrulanamadı. Lütfen tekrar deneyin.")
                    st.session_state["auth_nonce"] = _new_nonce()


def ensure_authenticated() -> bool:
    """Kullanıcı giriş yapmamışsa giriş ekranını gösterir. Giriş yapıldıysa True döner."""
    if "auth_nonce" not in st.session_state:
        st.session_state["auth_nonce"] = _new_nonce()
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["wallet"] = None

    if not st.session_state["authenticated"]:
        st.title("🔐 Giriş Gerekli")
        st.write(
            "Bu uygulamayı kullanmak için MetaMask cüzdanınızı bağlayıp "
            "aşağıdaki imza isteğini onaylamanız gerekiyor. İmza herhangi "
            "bir ücret gerektirmez ve işlem başlatmaz."
        )
        render_metamask_login()
        return False

    return True


# ==========================================================================
# 2) Parola tabanlı AES-256-GCM şifreleme
# ==========================================================================

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_message(password: str, plaintext: str) -> bytes:
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return salt + nonce + ciphertext  # ciphertext, GCM tag'ini de içerir


def decrypt_message(password: str, blob: bytes) -> str:
    salt, nonce, ciphertext = blob[:SALT_LEN], blob[SALT_LEN:SALT_LEN + NONCE_LEN], blob[SALT_LEN + NONCE_LEN:]
    key = derive_key(password, salt)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


# ==========================================================================
# 3) Parolayla karıştırılmış LSB steganografi
# ==========================================================================

def _scrambled_positions(password: str, total_positions: int, count: int) -> list[int]:
    """Parolaya bağlı bir PRNG ile [0, total_positions) aralığından `count`
    adet benzersiz, karıştırılmış konum üretir. Aynı parola her zaman aynı
    diziyi üretir; bu sayede gömme ve çözme birbiriyle tutarlı olur."""
    seed = int.from_bytes(hashlib.sha256(password.encode("utf-8")).digest(), "big")
    rng = random.Random(seed)
    positions = list(range(total_positions))
    rng.shuffle(positions)
    if count > total_positions:
        raise ValueError("Resim, bu kadar veriyi gizlemek için yeterli kapasiteye sahip değil.")
    return positions[:count]


def _bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        bits.extend((byte >> i) & 1 for i in range(7, -1, -1))
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        value = 0
        for b in byte_bits:
            value = (value << 1) | b
        out.append(value)
    return bytes(out)


def embed_data_in_image(image: Image.Image, password: str, plaintext: str) -> Image.Image:
    encrypted_blob = encrypt_message(password, plaintext)
    payload_len = len(encrypted_blob)

    rgb_image = image.convert("RGB")
    channel_array = np.array(rgb_image, dtype=np.uint8).flatten()
    total_positions = channel_array.size

    header_bits = [(payload_len >> i) & 1 for i in range(HEADER_BITS - 1, -1, -1)]
    payload_bits = _bytes_to_bits(encrypted_blob)
    all_bits = header_bits + payload_bits

    positions = _scrambled_positions(password, total_positions, len(all_bits))

    for pos, bit in zip(positions, all_bits):
        channel_array[pos] = (channel_array[pos] & 0xFE) | bit

    new_array = channel_array.reshape(np.array(rgb_image).shape)
    return Image.fromarray(new_array, mode="RGB")


def extract_data_from_image(image: Image.Image, password: str) -> str:
    rgb_image = image.convert("RGB")
    channel_array = np.array(rgb_image, dtype=np.uint8).flatten()
    total_positions = channel_array.size

    # Önce başlığı (uzunluk) okumak için HEADER_BITS kadar konum üretiyoruz
    header_positions = _scrambled_positions(password, total_positions, HEADER_BITS)
    header_bits = [int(channel_array[p] & 1) for p in header_positions]
    payload_len = 0
    for b in header_bits:
        payload_len = (payload_len << 1) | b

    if payload_len <= 0 or payload_len > total_positions:
        raise ValueError("Bu resimde geçerli bir gizli veri bulunamadı (ya da parola hatalı).")

    total_bits_needed = HEADER_BITS + payload_len * 8
    all_positions = _scrambled_positions(password, total_positions, total_bits_needed)
    payload_positions = all_positions[HEADER_BITS:]

    payload_bits = [int(channel_array[p] & 1) for p in payload_positions]
    encrypted_blob = _bits_to_bytes(payload_bits)

    try:
        return decrypt_message(password, encrypted_blob)
    except Exception:
        raise ValueError("Veri çözülemedi. Parola yanlış olabilir ya da resim bozulmuş olabilir.")


# ==========================================================================
# 4) Pinata / IPFS entegrasyonu
# ==========================================================================

def upload_to_pinata(image: Image.Image) -> Optional[str]:
    if not PINATA_API_KEY or not PINATA_SECRET_KEY:
        st.error("Pinata API anahtarları tanımlı değil (ortam değişkeni / secrets kontrol edin).")
        return None

    buffer = BytesIO()
    image.save(buffer, format="PNG")  # PNG zorunlu: kayıpsız olmalı

    files = {"file": ("stego_image.png", buffer.getvalue())}
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET_KEY,
    }
    try:
        response = requests.post(PINATA_ENDPOINT, files=files, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Pinata'ya yükleme sırasında hata oluştu: {exc}")
        return None
    return response.json().get("IpfsHash")


def fetch_from_pinata(ipfs_hash: str) -> Optional[Image.Image]:
    try:
        response = requests.get(f"{PINATA_GATEWAY_PREFIX}{ipfs_hash}", timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"IPFS'ten resim alınırken hata oluştu: {exc}")
        return None
    return Image.open(BytesIO(response.content))


# ==========================================================================
# 5) Streamlit ekranları
# ==========================================================================

def render_copy_button(text_to_copy: str, label: str = "📋 IPFS Kodunu Kopyala"):
    """Tek tıkla panoya kopyalama düğmesi (tarayıcının execCommand('copy')
    yöntemini kullanır; bu, MetaMask'ın aksine üst pencere iznine ihtiyaç
    duymadığı için Streamlit'in iframe'i içinde sorunsuz çalışır)."""
    safe_text = html_module.escape(text_to_copy)
    safe_label = html_module.escape(label)
    html_code = f"""
    <div style="font-family: sans-serif;">
      <input type="text" value="{safe_text}" id="copyInput" readonly
             style="position:absolute; left:-9999px;">
      <button onclick="copyToClipboard()" style="
          background:#4CAF50;color:white;border:none;padding:8px 18px;
          border-radius:6px;font-size:14px;cursor:pointer;">
        {safe_label}
      </button>
      <span id="copyStatus" style="margin-left:10px;color:#555;font-size:13px;"></span>
    </div>
    <script>
      function copyToClipboard() {{
        const input = document.getElementById('copyInput');
        input.select();
        input.setSelectionRange(0, 99999);
        try {{
          document.execCommand('copy');
          document.getElementById('copyStatus').innerText = 'Kopyalandı ✓';
        }} catch (err) {{
          document.getElementById('copyStatus').innerText = 'Kopyalanamadı, lütfen elle kopyalayın.';
        }}
      }}
    </script>
    """
    components.html(html_code, height=50)


def render_encode_screen():
    st.subheader("📥 Mesajı Resme Gizle")
    col_input, col_output = st.columns(2)

    file = col_input.file_uploader("Kapak resmi yükleyin (PNG önerilir)", type=["png", "jpg", "jpeg"])
    if not file:
        col_input.info("Lütfen bir resim yükleyin.")
        return

    image = Image.open(BytesIO(file.getvalue()))
    col_input.image(image, caption="Kapak resmi", use_container_width=True)

    capacity_bits = np.array(image.convert("RGB")).size
    max_payload_bytes = max((capacity_bits - HEADER_BITS) // 8 - (SALT_LEN + NONCE_LEN + 16), 0)
    col_input.caption(f"Yaklaşık maksimum mesaj kapasitesi: ~{max_payload_bytes} karakter")

    message = col_input.text_area("Gizlenecek mesaj")
    password = col_input.text_input("Şifreleme parolası", type="password",
                                     help="Bu parola olmadan mesaj çözülemez. Unutmayın!")

    if col_input.button("🔒 Şifrele, Gizle ve IPFS'e Yükle", type="primary"):
        if not message:
            col_input.error("Mesaj boş olamaz.")
            return
        if not password:
            col_input.error("Lütfen bir parola girin.")
            return

        try:
            with st.spinner("Mesaj şifreleniyor ve resme gömülüyor..."):
                stego_image = embed_data_in_image(image, password, message)
        except ValueError as exc:
            col_input.error(str(exc))
            return

        with st.spinner("IPFS'e yükleniyor..."):
            ipfs_hash = upload_to_pinata(stego_image)

        if not ipfs_hash:
            col_output.error("Resim IPFS'e yüklenirken bir hata oluştu.")
            return

        log_action(st.session_state.get("wallet", ""), "encode", ipfs_hash, success=True)

        col_output.success("Mesaj başarıyla gizlendi ve IPFS'e yüklendi.")
        col_output.code(ipfs_hash, language=None)
        with col_output:
            render_copy_button(ipfs_hash)


def render_decode_screen():
    st.subheader("📤 Resimden Mesajı Çöz")
    ipfs_hash = st.text_input("IPFS Hash")
    password = st.text_input("Şifreleme parolası", type="password")

    if st.button("🔓 Çöz", type="primary"):
        if not ipfs_hash or not password:
            st.error("IPFS hash ve parola gereklidir.")
            return

        with st.spinner("Resim IPFS'ten indiriliyor..."):
            image = fetch_from_pinata(ipfs_hash)
        if image is None:
            st.error("Hatalı IPFS hash. Lütfen doğru kodu girdiğinizden emin olun.")
            log_action(st.session_state.get("wallet", ""), "decode", ipfs_hash, success=False)
            return

        try:
            with st.spinner("Veri çözülüyor..."):
                message = extract_data_from_image(image, password)
        except ValueError as exc:
            st.error(str(exc))
            log_action(st.session_state.get("wallet", ""), "decode", ipfs_hash, success=False)
            return

        log_action(st.session_state.get("wallet", ""), "decode", ipfs_hash, success=True)
        st.success("Mesaj başarıyla çözüldü:")
        st.text_area("Çözülen mesaj", value=message, height=150)
        st.image(image, caption="Kaynak resim", use_container_width=True)


def render_sidebar_account():
    wallet = st.session_state.get("wallet", "")
    label = "👑 Yönetici" if is_admin_wallet(wallet) else "Bağlı cüzdan"
    st.sidebar.success(f"{label}:\n`{wallet}`")
    if st.sidebar.button("Çıkış yap"):
        for key in ("authenticated", "wallet", "auth_nonce", "login_attempt"):
            st.session_state.pop(key, None)
        st.rerun()


def render_admin_panel():
    st.subheader("👑 Yönetici Paneli")

    users = list_users()
    st.markdown(f"**Toplam kullanıcı:** {len(users)}")

    st.markdown("### Kullanıcılar")
    for user in users:
        wallet = user["wallet_address"]
        is_self_admin = is_admin_wallet(wallet)
        cols = st.columns([3, 2, 2, 2])
        cols[0].code(wallet, language=None)
        cols[1].caption("İlk görülme: " + time.strftime("%Y-%m-%d %H:%M", time.localtime(user["first_seen"])))
        cols[2].caption("Son görülme: " + time.strftime("%Y-%m-%d %H:%M", time.localtime(user["last_seen"])))

        if is_self_admin:
            cols[3].caption("👑 Yönetici (engellenemez)")
        elif user["is_banned"]:
            if cols[3].button("Engeli Kaldır", key=f"unban_{wallet}"):
                set_wallet_banned(wallet, False)
                st.rerun()
        else:
            if cols[3].button("Engelle", key=f"ban_{wallet}"):
                set_wallet_banned(wallet, True, reason="Yönetici tarafından engellendi")
                st.rerun()

    st.divider()
    st.markdown("### Son İşlemler")
    activity = list_recent_activity(limit=100)
    if not activity:
        st.caption("Henüz işlem kaydı yok.")
    else:
        st.dataframe(
            [
                {
                    "Zaman": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a["timestamp"])),
                    "Cüzdan": a["wallet_address"],
                    "İşlem": a["action"],
                    "IPFS Hash": a["ipfs_hash"] or "-",
                    "Başarılı mı": "✅" if a["success"] else "❌",
                }
                for a in activity
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Not: Buradan yalnızca kim, ne zaman, hangi işlemi yaptığı görülebilir. "
        "Parolalar ve gizlenen mesaj içerikleri hiçbir zaman kaydedilmez, "
        "bu yüzden burada da görüntülenemez."
    )


def main():
    if not ensure_authenticated():
        return

    wallet = st.session_state.get("wallet", "")

    # Oturum devam ederken yönetici kullanıcıyı sonradan engellemiş olabilir;
    # her sayfa yenilemesinde tekrar kontrol edip gerekirse oturumu kapatıyoruz.
    if is_wallet_banned(wallet):
        st.error("Bu cüzdan yönetici tarafından engellenmiştir. Oturumunuz kapatılıyor.")
        for key in ("authenticated", "wallet", "auth_nonce", "login_attempt"):
            st.session_state.pop(key, None)
        st.stop()

    st.title("🔐 Steganografi Bilimine Yeni Bir Boyut")
    render_sidebar_account()

    tab_options = ["Mesaj Gizle", "Mesaj Çöz"]
    if is_admin_wallet(wallet):
        tab_options.append("Yönetici Paneli")

    tab = st.sidebar.radio("İşlem seçin", tab_options)
    if tab == "Mesaj Gizle":
        render_encode_screen()
    elif tab == "Mesaj Çöz":
        render_decode_screen()
    else:
        render_admin_panel()


if __name__ == "__main__":
    main()