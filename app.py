# -*- coding: utf-8 -*-
"""Giao dien Streamlit: goi y khach san (Yeu cau 1) + collaborative filtering (Yeu cau 2)
+ insight khach san (Yeu cau 3).

3 tab, dung khung 7 tang giong app.py xe may:
  1. Cau hinh & hang so
  2. Tai model / du lieu (cache)
  3. Ham tien ich hien thi
  4. Ham thu thap input
  5. Tien xu ly van ban / tokenize
  6. Ham nghiep vu thuan (khong co st.*)
  7. UI

Yeu cau 1 — Content-based filtering (khop Project1_Request1.ipynb):
  - Tim theo mo ta tu do: TfidfVectorizer + cosine_similarity (search_cosine),
    uu tien khach san diem cao (Total_Score) trong tap ket qua lien quan.

Yeu cau 2 — Collaborative filtering (khop Project1_Request2.ipynb):
  - Item-Based KNN (NearestNeighbors, cosine) tren ma tran Hotel x User.

Yeu cau 3 — Insight cho chu khach san (khop Project1_Request3.ipynb):
  - Tong quan, diem manh/yeu, thong ke khach hang, tu khoa noi bat,
    so sanh he thong, va vai insight tu dong rut ra.
"""
from __future__ import annotations

import os
import re
import textwrap
import urllib.parse
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================================
# TANG 1 — CAU HINH & HANG SO
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
HOTEL_PHOTOS_DIR = os.path.join(ASSETS_DIR, "hotel_photos")
HOTEL_PHOTOS_XLSX = os.path.join(HOTEL_PHOTOS_DIR, "photos.xlsx")

HOTEL_PATH = os.path.join(DATA_DIR, "hotel_info_clean.pkl")
COMMENT_PATH = os.path.join(DATA_DIR, "hotel_comments_clean.pkl")

DICTIONARY_PATH = os.path.join(MODEL_DIR, "dictionary.pkl")
TFIDF_GENSIM_PATH = os.path.join(MODEL_DIR, "tfidf_gensim.pkl")
GENSIM_INDEX_PATH = os.path.join(MODEL_DIR, "sparse_index.pkl")
GENSIM_SIM_PATH = os.path.join(MODEL_DIR, "gensim_sim.npy")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
COSINE_SIM_PATH = os.path.join(MODEL_DIR, "cosine_sim.npy")
TEXT_RES_PATH = os.path.join(MODEL_DIR, "text_resources.joblib")
EVAL_PATH = os.path.join(MODEL_DIR, "eval_metrics.joblib")

CF_MATRIX_PATH = os.path.join(MODEL_DIR, "cf_matrix.npz")
CF_KNN_PATH = os.path.join(MODEL_DIR, "cf_knn_model.joblib")
CF_HOTEL_INDEX_PATH = os.path.join(MODEL_DIR, "cf_hotel_index.pkl")

# Ten cot — khop voi Project1_Request1.ipynb + Project1_Request3.ipynb
C_ID = "Hotel_ID"
C_NAME = "Hotel_Name"
C_RANK = "Hotel_Rank"
C_ADDR = "Hotel_Address"
C_SCORE = "Total_Score"
C_DESC = "Hotel_Description"
C_CONTENT = "Content_clean"
C_COUNT = "comments_count"

# 6 tieu chi diem chi tiet — khop score_cols trong Project1_Request3.ipynb
SCORE_COLS = ["Location", "Cleanliness", "Service",
              "Facilities", "Value_for_money", "Comfort_and_room_quality"]

CM_ID = "Hotel_ID"       # cot Hotel_ID trong hotel_comments (sau khi chuan hoa ten cot)
CM_SCORE = "Score"
CM_BODY = "Body"
CM_NATION = "Nationality"
CM_GROUP = "Group_Name"
CM_YEARMONTH = "Review_YearMonth"

POSITIVE_THRESHOLD = 8    # danh gia >= 8 diem: hai long (khop notebook)
NEGATIVE_THRESHOLD = 6    # danh gia < 6 diem: khong hai long (khop notebook)
TOP_N_DEFAULT = 5
TOP_KEYWORDS = 15
MIN_PIE_PCT = 5           # lat pie < 5% se duoc gop vao nhom "Khac" cho de nhin

# Stopword rieng cho phan phan tich tu khoa insight — khop Project1_Request3.ipynb
VIETNAMESE_STOPWORDS = set("""
và là của có cho được rất đã rồi thì mà không những các một hai ba trong khi
tôi chúng tôi bạn họ mình này đó nên nếu vì do bởi để nhưng cũng còn nữa như
với từ tại nơi khi nào ở trên dưới ra vào lên xuống đi lại qua sẽ đang bị
những là các một số nhiều ít mỗi từng cả tất cả rất là quá khá hơi hơn nhất
mình tôi khách sạn phòng ks nhân viên lần đêm ngày lúc giờ
""".split())

ENGLISH_STOPWORDS_INSIGHT = set("""
the and was with for is in to of this that we our very are at on as it had has
an but not all from by were their they you your when what which more most
just there here also can could would should will been being do does did have
""".split())

GENERIC_WORDS = {"hotel", "room", "khach", "san"}
ALL_STOPWORDS_INSIGHT = VIETNAMESE_STOPWORDS | ENGLISH_STOPWORDS_INSIGHT | GENERIC_WORDS

st.set_page_config(page_title="Hệ thống khách sạn Nha Trang", page_icon="🏨", layout="wide")


# =========================================================================
# TANG 2 — TAI MODEL / DU LIEU (CACHE)
# =========================================================================
@st.cache_resource
def load_content_model():
    """Nap toan bo artifact content-based; key nao thieu file thi = None."""
    art = {}
    art["gensim_sim"] = np.load(GENSIM_SIM_PATH) if os.path.exists(GENSIM_SIM_PATH) else None
    art["cosine_sim"] = np.load(COSINE_SIM_PATH) if os.path.exists(COSINE_SIM_PATH) else None

    art["dictionary"] = art["tfidf_gensim"] = art["index_gensim"] = None
    if all(os.path.exists(p) for p in (DICTIONARY_PATH, TFIDF_GENSIM_PATH, GENSIM_INDEX_PATH)):
        from gensim import corpora, models, similarities
        art["dictionary"] = corpora.Dictionary.load(DICTIONARY_PATH)
        art["tfidf_gensim"] = models.TfidfModel.load(TFIDF_GENSIM_PATH)
        art["index_gensim"] = similarities.SparseMatrixSimilarity.load(GENSIM_INDEX_PATH)

    art["vectorizer"] = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None
    if os.path.exists(TFIDF_MATRIX_PATH):
        from scipy.sparse import load_npz
        art["tfidf_matrix"] = load_npz(TFIDF_MATRIX_PATH)
    else:
        art["tfidf_matrix"] = None

    art["text_res"] = joblib.load(TEXT_RES_PATH) if os.path.exists(TEXT_RES_PATH) else {}
    art["eval"] = joblib.load(EVAL_PATH) if os.path.exists(EVAL_PATH) else None
    return art


@st.cache_resource
def load_cf_model():
    """
    Nap model Item-Based KNN (Yeu cau 2) — khop Project1_Request2.ipynb.
    Tra ve dict co "knn", "matrix", "hotel_index" deu None neu chua build.
    """
    art = {"knn": None, "matrix": None, "hotel_index": None}
    if os.path.exists(CF_KNN_PATH):
        art["knn"] = joblib.load(CF_KNN_PATH)
    if os.path.exists(CF_MATRIX_PATH):
        from scipy.sparse import load_npz
        art["matrix"] = load_npz(CF_MATRIX_PATH)
    if os.path.exists(CF_HOTEL_INDEX_PATH):
        art["hotel_index"] = pd.read_pickle(CF_HOTEL_INDEX_PATH)
    return art


@st.cache_data
def load_hotels() -> pd.DataFrame:
    return pd.read_pickle(HOTEL_PATH)


@st.cache_data
def load_comments() -> pd.DataFrame:
    return pd.read_pickle(COMMENT_PATH) if os.path.exists(COMMENT_PATH) else pd.DataFrame()


@st.cache_data
def load_hotel_photos() -> dict:
    """
    Doc danh sach anh minh hoa (do NGUOI DUNG tu chuan bi, khong phai anh that cua tung khach san)
    tu assets/hotel_photos/photos.xlsx, encode base64 tung anh hop le.
    Tra ve {} neu chua co file Excel hoac chua co anh nao dung duoc — luc do UI se tu lui ve
    dung anh minh hoa SVG ve tay (xem hotel_placeholder_svg).
    """
    import base64
    if not os.path.exists(HOTEL_PHOTOS_XLSX):
        return {}
    try:
        df = pd.read_excel(HOTEL_PHOTOS_XLSX, sheet_name="Anh_minh_hoa", header=7)
    except Exception:
        return {}

    photos = {}
    for _, row in df.iterrows():
        fname = str(row.get("Ten_file (tên file ảnh)", "")).strip()
        su_dung = str(row.get("Su_dung (TRUE/FALSE)", "TRUE")).strip().upper()
        if not fname or fname == "nan" or su_dung == "FALSE":
            continue
        fpath = os.path.join(HOTEL_PHOTOS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            photos[fname] = base64.b64encode(f.read()).decode("ascii")
    return photos


@st.cache_data
def compute_system_avg(df_hotel: pd.DataFrame) -> pd.Series:
    """Diem trung binh toan he thong cho tung tieu chi — dung lam moc so sanh (khop notebook)."""
    cols = [c for c in ([C_SCORE] + SCORE_COLS) if c in df_hotel.columns]
    return df_hotel[cols].mean()


# =========================================================================
# TANG 3 — HAM TIEN ICH HIEN THI
# =========================================================================
def fmt_score(x) -> str:
    if pd.isna(x):
        return "—"
    try:
        return f"{float(str(x).strip().replace(',', '.')):.1f}"
    except (ValueError, TypeError):
        return "—"


# Bang mau (dung de phoi mau cho tung minh hoa) — theo dung tong mau hero
PLACEHOLDER_PALETTES = [
    ("#073B4C", "#0E7C86"),   # dem sau
    ("#0E7C86", "#2EC4B6"),   # ngoc bich
    ("#164FA3", "#1D6FE0"),   # xanh duong
    ("#B5541F", "#FF6F59"),   # hoang hon
    ("#0E7C86", "#F4E3C1"),   # cat & bien
]


def _scene_beach_palm(c1, c2, sand):
    """Bien & la dua — goc phai tren co la dua ruong xuong."""
    return f"""<rect width="100" height="100" rx="14" fill="url(#g)"/>
<rect x="0" y="70" width="100" height="30" rx="0" fill="{sand}"/>
<circle cx="80" cy="20" r="10" fill="#FDFBF7" opacity="0.85"/>
<path d="M100,10 Q75,15 65,35 Q80,25 90,15 Q95,25 85,40 Q95,32 100,25 Z" fill="#0E7C86" opacity="0.9"/>"""


def _scene_pool_top(c1, c2, sand):
    """Ho boi nhin tu tren — hinh oval xanh + vien trang."""
    return f"""<rect width="100" height="100" rx="14" fill="{sand}"/>
<ellipse cx="50" cy="52" rx="40" ry="30" fill="url(#g)"/>
<ellipse cx="50" cy="52" rx="40" ry="30" fill="none" stroke="#FDFBF7" stroke-width="3"/>
<ellipse cx="38" cy="42" rx="10" ry="6" fill="#FDFBF7" opacity="0.35"/>"""


def _scene_sunset_roof(c1, c2, dark):
    """Hoang hon san thuong — mat troi to thap + silhouette toa nha."""
    return f"""<rect width="100" height="100" rx="14" fill="url(#g)"/>
<circle cx="50" cy="62" r="20" fill="#FDFBF7" opacity="0.9"/>
<rect x="0" y="78" width="100" height="22" fill="{dark}"/>
<rect x="14" y="66" width="10" height="14" fill="{dark}"/>
<rect x="30" y="70" width="8" height="10" fill="{dark}"/>
<rect x="62" y="68" width="9" height="12" fill="{dark}"/>
<rect x="78" y="72" width="8" height="8" fill="{dark}"/>"""


def _scene_window_view(c1, c2, frame):
    """Cua so nhin ra bien — khung cua + bien gradient ben trong."""
    return f"""<rect width="100" height="100" rx="14" fill="{frame}"/>
<rect x="12" y="12" width="76" height="76" rx="6" fill="url(#g)"/>
<rect x="12" y="46" width="76" height="4" fill="{frame}" opacity="0.5"/>
<rect x="46" y="12" width="4" height="76" fill="{frame}" opacity="0.5"/>
<circle cx="70" cy="28" r="7" fill="#FDFBF7" opacity="0.85"/>"""


def _scene_pier(c1, c2, wood):
    """Ben du thuyen — duong chan troi + cot go nho."""
    return f"""<rect width="100" height="100" rx="14" fill="url(#g)"/>
<rect x="0" y="66" width="100" height="4" fill="#FDFBF7" opacity="0.6"/>
<rect x="20" y="66" width="5" height="30" fill="{wood}"/>
<rect x="45" y="66" width="5" height="30" fill="{wood}"/>
<rect x="70" y="66" width="5" height="30" fill="{wood}"/>
<circle cx="76" cy="22" r="9" fill="#FDFBF7" opacity="0.85"/>"""


def _scene_fishing_village(c1, c2, sand):
    """Xom chai buoi sang — day nha nho ven bien."""
    return f"""<rect width="100" height="100" rx="14" fill="url(#g)"/>
<rect x="0" y="74" width="100" height="26" fill="{sand}"/>
<polygon points="10,74 20,60 30,74" fill="#FDFBF7" opacity="0.9"/>
<rect x="12" y="74" width="16" height="1" fill="{sand}"/>
<polygon points="38,74 47,62 56,74" fill="#0E7C86" opacity="0.85"/>
<polygon points="62,74 73,58 84,74" fill="#FDFBF7" opacity="0.75"/>"""


_SCENES = [_scene_beach_palm, _scene_pool_top, _scene_sunset_roof,
           _scene_window_view, _scene_pier, _scene_fishing_village]


def render_scene_svg(scene_fn, c1, c2, gid: str, corner_label: str = "") -> str:
    """Ve 1 canh minh hoa (dung chung cho anh dai dien khach san va dai goi y nhanh)."""
    accent = "#F4E3C1" if scene_fn in (_scene_beach_palm, _scene_pool_top, _scene_fishing_village) else "#073B4C"
    body = scene_fn(c1, c2, accent)
    label_svg = (
        f'<text x="93" y="93" font-family="\'Public Sans\', sans-serif" font-size="13" '
        f'font-weight="700" fill="#FDFBF7" text-anchor="end" opacity="0.55">{corner_label}</text>'
        if corner_label else ""
    )
    return f"""<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ảnh minh hoạ">
<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/>
</linearGradient></defs>
<g>{body.replace('url(#g)', f'url(#{gid})')}</g>
{label_svg}
</svg>"""


def hotel_placeholder_svg(hotel_id, name: str) -> str:
    """
    Anh dai dien minh hoa (khong phai anh that) cho khach san — vi du lieu chua co anh.
    100% ve tay bang SVG (khong tai/nhung anh cua ai), chon 1 trong 6 kieu + 1 tong mau
    on dinh theo hotel_id (dung ma hoa, khong dung hash() vi no doi moi lan chay).
    """
    import hashlib
    digest = hashlib.md5(str(hotel_id).encode("utf-8")).hexdigest()
    n = int(digest, 16)
    c1, c2 = PLACEHOLDER_PALETTES[n % len(PLACEHOLDER_PALETTES)]
    scene_fn = _SCENES[n % len(_SCENES)]
    letter = (str(name).strip()[:1] or "?").upper()
    return render_scene_svg(scene_fn, c1, c2, gid=f"g{digest[:8]}", corner_label=letter)


def hotel_thumb_html(hotel_id, name: str, photos: dict) -> str:
    """
    Chon anh dai dien cho 1 khach san: uu tien anh that (do nguoi dung tu chuan bi trong
    assets/hotel_photos/), chon on dinh theo hotel_id (cung 1 khach san luon ra cung 1 anh,
    khong doi ngau nhien moi lan chay lai). Neu chua co anh nao -> lui ve SVG minh hoa ve tay.
    """
    if photos:
        import hashlib
        names = sorted(photos.keys())
        idx = int(hashlib.md5(str(hotel_id).encode("utf-8")).hexdigest(), 16) % len(names)
        b64 = photos[names[idx]]
        return f'<img src="data:image/jpeg;base64,{b64}" alt="Ảnh minh hoạ" loading="lazy"/>'
    return hotel_placeholder_svg(hotel_id, name)


def render_comparison_table(comparison: pd.DataFrame):
    """
    Bang so sanh Khach san / Trung binh he thong / Chenh lech — dung HTML tuy chinh
    thay vi st.dataframe de dam bao tieu de in dam that su (st.dataframe khong cho
    chinh font-weight cua header), va to mau o Chenh lech theo dau (+/-).
    """
    rows_html = []
    for crit, row in comparison.round(2).iterrows():
        hotel_val, sys_val, diff = row["Khách sạn"], row["Trung bình hệ thống"], row["Chênh lệch"]
        hotel_str = "—" if pd.isna(hotel_val) else f"{hotel_val:.2f}"
        sys_str = "—" if pd.isna(sys_val) else f"{sys_val:.2f}"
        if pd.isna(diff):
            diff_str, diff_color = "—", "#888888"
        else:
            diff_color = "#0E7C86" if diff >= 0 else "#E74C3C"
            diff_str = f"{'+' if diff > 0 else ''}{diff:.2f}"
        rows_html.append(f"""<tr>
<td class="cmp-crit">{crit}</td>
<td class="cmp-num">{hotel_str}</td>
<td class="cmp-num">{sys_str}</td>
<td class="cmp-num" style="color:{diff_color}; font-weight:700;">{diff_str}</td>
</tr>""")

    html = f"""
<style>
.cmp-table {{ width: 100%; border-collapse: collapse; font-family: 'Public Sans', sans-serif; font-size: 0.92rem; }}
.cmp-table th {{
    text-align: right; font-weight: 700; color: var(--deepsea);
    background: var(--sand); padding: 0.55rem 0.8rem; border-bottom: 2px solid var(--deepsea);
}}
.cmp-table th:first-child {{ text-align: left; }}
.cmp-table td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #ECECEC; }}
.cmp-table td.cmp-crit {{ text-align: left; }}
.cmp-table td.cmp-num {{ text-align: right; }}
.cmp-table tr:last-child td {{ border-bottom: none; }}
</style>
<table class="cmp-table">
<thead><tr><th>Tiêu chí</th><th>Khách sạn</th><th>Trung bình hệ thống</th><th>Chênh lệch</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_hotel_detail_page(hotel_id, df_hotel: pd.DataFrame, df_cmt: pd.DataFrame,
                              photos: dict, state_key: str):
    """
    Trang chi tiet khach san (thay the noi dung CHINH CUA DUNG TAB dang xem, khong phai
    popup, va KHONG an thanh tab di) — anh, dia chi, tong diem, so luot danh gia, mo ta,
    va DANH SACH BINH LUAN THAT cua khach de nguoi xem doc thu — giong Agoda.

    Luu y ky thuat: co tinh KHONG bao gio bo qua goi st.tabs() o tang tren — Streamlit
    se "quen" tab dang chon neu widget tabs khong duoc goi lien tuc vai luot chay.
    Vi vay trang chi tiet nay duoc ve NGAY BEN TRONG than cua dung tab da bam "Xem chi
    tiet", thay vi thay the toan bo trang — nho vay tab luon duoc goi, khong bao gio mat
    trang thai, va nguoi dung tu nhien o lai dung tab cu sau khi bam Quay lai.
    """
    if st.button("← Quay lại danh sách", key=f"back_{state_key}"):
        del st.session_state[state_key]
        st.rerun()

    row_match = df_hotel[df_hotel[C_ID] == hotel_id]
    if row_match.empty:
        st.error("Không tìm thấy khách sạn này.")
        return
    row = row_match.iloc[0]

    thumb_html = hotel_thumb_html(hotel_id, row.get(C_NAME, ""), photos)
    st.markdown(f'<div class="hotel-detail-thumb">{thumb_html}</div>', unsafe_allow_html=True)

    st.title(row.get(C_NAME, ""))
    address_text = str(row.get(C_ADDR, "") or "")
    maps_url = "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(address_text)
    st.markdown(
        f'<div class="hotel-detail-meta">'
        f'<span class="meta-star">⭐</span> {row.get(C_RANK, "")}'
        f' &nbsp;·&nbsp; '
        f'<span class="meta-pin">📍</span> '
        f'<a href="{maps_url}" target="_blank" class="meta-map-link">{address_text}</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Tổng điểm", f"{fmt_score(row.get(C_SCORE))}/10")
    count = row.get(C_COUNT, 0)
    c2.metric("Số lượt đánh giá", f"{int(count) if pd.notna(count) else 0:,}".replace(",", "."))

    desc = str(row.get(C_DESC, "") or "")
    if desc:
        st.divider()
        st.write(desc)

    st.divider()
    st.subheader("💬 Đánh giá từ khách")
    reviews = df_cmt[df_cmt[CM_ID] == hotel_id] if not df_cmt.empty and CM_ID in df_cmt.columns else pd.DataFrame()
    if reviews.empty:
        st.caption("Khách sạn này chưa có đánh giá nào.")
        return

    # Moi khach san co bo dem rieng — de khong bi "nho nham" so luong tu khach san truoc
    shown_key = f"reviews_shown_{hotel_id}"
    if shown_key not in st.session_state:
        st.session_state[shown_key] = 10
    show_n = st.session_state[shown_key]

    reviews_sorted = reviews.sort_values(CM_SCORE, ascending=False, na_position="last")
    st.caption(f"Hiển thị {min(show_n, len(reviews))} / {len(reviews)} đánh giá — điểm cao nhất trước.")

    for _, rv in reviews_sorted.head(show_n).iterrows():
        with st.container(border=True):
            meta_bits = []
            reviewer = rv.get("Reviewer_Name")
            if pd.notna(reviewer):
                meta_bits.append(f"**{reviewer}**")
            nation = rv.get(CM_NATION)
            if pd.notna(nation):
                meta_bits.append(str(nation))
            group = rv.get(CM_GROUP)
            if pd.notna(group):
                meta_bits.append(str(group))
            room = rv.get("Room_Type")
            if pd.notna(room):
                meta_bits.append(str(room))
            if meta_bits:
                st.caption(" · ".join(meta_bits))

            score = rv.get(CM_SCORE)
            if pd.notna(score):
                st.markdown(f"**⭐ {float(score):.1f}/10**")

            title = rv.get("Title")
            if pd.notna(title) and str(title).strip():
                st.markdown(f"*{title}*")

            body = rv.get(CM_BODY)
            if pd.notna(body) and str(body).strip():
                st.write(body)

    if show_n < len(reviews):
        if st.button(f"Xem thêm đánh giá ({len(reviews) - show_n} còn lại)", key=f"more_{hotel_id}"):
            st.session_state[shown_key] += 10
            st.rerun()


def render_results(df: pd.DataFrame, photos: dict = None, state_key: str = "viewing_hotel_id"):
    """
    Hien thi danh sach khach san goi y — the gon, anh dai dien, ten in dam.
    state_key: khoa session_state rieng cho tung tab (VD "tab1_viewing_id",
    "tab2_viewing_id") — de bam Xem chi tiet o tab nao thi chi tab do doi giao dien,
    khong lam mat trang thai cua thanh tab (xem giai thich trong render_hotel_detail_page).
    """
    photos = photos or {}
    if df.empty:
        st.warning("Không tìm thấy khách sạn phù hợp. Thử đổi khách sạn khác hoặc mô tả khác.")
        return
    for _, r in df.iterrows():
        with st.container(border=True, key=f"result_{state_key}_{r.get(C_ID)}"):
            c_img, c_info, c_score = st.columns([1, 5, 1.3], vertical_alignment="center")
            with c_img:
                st.markdown(
                    f'<div class="hotel-thumb">{hotel_thumb_html(r.get(C_ID), r.get(C_NAME), photos)}</div>',
                    unsafe_allow_html=True,
                )
            with c_info:
                st.markdown(f'<div class="hotel-card-title">{r.get(C_NAME, "")}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="hotel-detail-meta hotel-card-meta">'
                    f'<span class="meta-star">⭐</span> {r.get(C_RANK, "")}'
                    f' &nbsp;·&nbsp; '
                    f'<span class="meta-pin">📍</span> {r.get(C_ADDR, "")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                desc = str(r.get(C_DESC, "") or "")
                if desc:
                    snippet = desc[:110] + ("…" if len(desc) > 110 else "")
                    st.caption(snippet)
                if st.button("👁 Xem chi tiết", key=f"viewdetail_{state_key}_{r.get(C_ID)}"):
                    st.session_state[state_key] = r.get(C_ID)
                    st.rerun()
            with c_score:
                st.metric("Điểm", fmt_score(r.get(C_SCORE)))
                st.caption(f"Tương đồng **{r['similarity']:.2f}**")




# =========================================================================
# TANG 4 — THU THAP INPUT
# =========================================================================
def render_search_inputs():
    """O nhap Yeu cau 1 — chi con 1 cach tim: theo mo ta noi dung (Cosine similarity)."""
    query = st.text_area(
        "Bạn muốn tìm khách sạn thế nào?",
        placeholder="VD: Khách sạn mới, phòng ngủ rộng, gần biển và thoáng mát, tiện nghi",
        height=90, key="query",
    )
    top_n = st.slider("Số khách sạn gợi ý", 3, 20, TOP_N_DEFAULT, key="top_n")
    return query, top_n


# (label hien thi, mo ta dien san, ham ve canh minh hoa, cap mau, slug ten file anh)
QUICK_SUGGESTIONS = [
    ("Gần biển, hàng dừa", "Khách sạn gần biển, có hàng dừa, không gian yên tĩnh", _scene_beach_palm, 0, "beach_palm"),
    ("Hồ bơi ấn tượng", "Khách sạn có hồ bơi đẹp, nhìn từ trên cao ấn tượng", _scene_pool_top, 1, "pool_top"),
    ("Hoàng hôn sân thượng", "Khách sạn có sân thượng ngắm hoàng hôn, quầy bar trên cao", _scene_sunset_roof, 3, "sunset_roof"),
    ("Phòng view biển", "Phòng có cửa sổ lớn nhìn ra biển, ánh sáng tự nhiên", _scene_window_view, 2, "window_view"),
    ("Gần bến du thuyền", "Khách sạn gần bến du thuyền, thuận tiện di chuyển", _scene_pier, 4, "pier"),
    ("Trải nghiệm xóm chài", "Gần khu xóm chài, trải nghiệm văn hoá địa phương", _scene_fishing_village, 0, "fishing_village"),
]

QUICK_SUGGESTIONS_DIR = os.path.join(ASSETS_DIR, "quick_suggestions")


@st.cache_data
def load_quick_suggestion_photo(slug: str):
    """
    Tim anh that cho 1 o goi y nhanh trong assets/quick_suggestions/{slug}.(jpg|jpeg|png|webp).
    Neu co: resize vuong 640x640 + encode base64. Neu khong tim thay file nao -> (None, None).
    Neu tim thay file nhung KHONG doc duoc (anh hong/sai dinh dang thuc su ben trong)
    -> (None, thong_bao_loi) de UI hien canh bao ro rang thay vi im lang lui ve SVG.
    """
    import base64
    import io

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        fname = f"{slug}{ext}"
        fpath = os.path.join(QUICK_SUGGESTIONS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            from PIL import Image, ImageOps
            im = Image.open(fpath)
            im.load()  # bat loi giai ma that su ngay tai day (Image.open chi doc header)
            im = im.convert("RGB")
            im = ImageOps.fit(im, (640, 640), method=Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii"), None
        except Exception as exc:
            return None, f"Không đọc được `{fname}`: {exc}"
    return None, None


def _apply_quick_suggestion(query_text: str):
    st.session_state["query"] = query_text


def render_quick_suggestions():
    """Dai goi y nhanh — hien khi chua tim gi, do trang khong bi trong khi moi vao web."""
    st.caption("✨ Gợi ý nhanh — bấm để tìm khách sạn theo phong cách bạn thích")
    cols = st.columns(len(QUICK_SUGGESTIONS))
    for col, (label, query_text, scene_fn, palette_idx, slug) in zip(cols, QUICK_SUGGESTIONS):
        photo_b64, err = load_quick_suggestion_photo(slug)
        if photo_b64:
            thumb_html = f'<img src="data:image/jpeg;base64,{photo_b64}" alt="{label}" loading="lazy"/>'
        else:
            c1, c2 = PLACEHOLDER_PALETTES[palette_idx]
            thumb_html = render_scene_svg(scene_fn, c1, c2, gid=f"sugg-{palette_idx}-{scene_fn.__name__}")
        with col:
            st.markdown(f'<div class="hotel-thumb">{thumb_html}</div>', unsafe_allow_html=True)
            st.button(label, key=f"sugg_{scene_fn.__name__}", use_container_width=True,
                      on_click=_apply_quick_suggestion, args=(query_text,))
            if err:
                st.caption(f"⚠️ {err}")


def render_hotel_picker(df_hotel: pd.DataFrame, key: str):
    """O chon khach san cho Yeu cau 3."""
    names = df_hotel[C_NAME].tolist()
    picked_name = st.selectbox("Chọn khách sạn của bạn", names, key=key)
    return df_hotel.loc[df_hotel[C_NAME] == picked_name, C_ID].iloc[0]


# =========================================================================
# TANG 5 — TIEN XU LY VAN BAN / TOKENIZE
# =========================================================================
def preprocess_text(text: str, res: dict) -> list:
    """
    Tien xu ly 1 cau search (Yeu cau 1) — PHAI giong het clean_content_column trong notebook:
    tokenize (pyvi) -> xoa so -> lowercase + bo ky tu dac biet -> bo stopword -> dich Anh->Viet.
    """
    from pyvi.ViTokenizer import tokenize

    special_chars = res.get("special_chars", [])
    all_stopwords = res.get("all_stopwords", set())
    en_vn_dict = res.get("en_vn_dict", {})
    english_stopwords = res.get("english_stopwords", set())

    words = tokenize(str(text)).split()
    words = [re.sub(r"[0-9]+", "", w) for w in words]
    words = [re.sub(r"[^\w_]", "", w.lower()) for w in words if w not in special_chars]
    words = [w for w in words if w not in all_stopwords and len(w) > 1]

    out = []
    for w in words:
        if w.lower() in english_stopwords:
            continue
        out.append(en_vn_dict.get(w.lower(), w))
    return out


def tokenize_vi(text) -> list:
    """
    Tokenize don gian cho phan tich tu khoa insight (Yeu cau 3) — khop tokenize_vi trong
    Project1_Request3.ipynb: lowercase, bo dau cau (giu dau tieng Viet), tach theo khoang trang.
    Khac voi preprocess_text ben tren: khong dung pyvi, khong dich Anh->Viet.
    """
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ]", " ", text,
    )
    return [w for w in text.split() if len(w) >= 3 and w not in ALL_STOPWORDS_INSIGHT]


# =========================================================================
# TANG 6 — HAM NGHIEP VU THUAN (khong co st.*)
# =========================================================================
# ---- Yeu cau 1: Content-based ----
def get_recommendations_gensim(df_hotel: pd.DataFrame, art: dict, hotel_id, nums=5) -> pd.DataFrame:
    """Khop get_recommendations_gensim trong notebook — tra bang gensim_sim da tinh san."""
    match = np.flatnonzero((df_hotel[C_ID] == hotel_id).to_numpy())
    if len(match) == 0 or art["gensim_sim"] is None:
        return pd.DataFrame()
    idx = int(match[0])

    row = np.asarray(art["gensim_sim"][idx], dtype=float).copy()
    row[idx] = -np.inf
    top_idx = np.argsort(-row)[:nums]

    out = df_hotel.iloc[top_idx].copy()
    out["similarity"] = row[top_idx]
    return out


def rank_by_score_with_fallback(df: pd.DataFrame, nums: int) -> pd.DataFrame:
    """
    Uu tien khach san co diem (Total_Score) tu cao xuong thap trong so cac ket qua
    lien quan (da qua loc do tuong dong noi dung). Thu bo cac khach san KHONG co diem
    truoc; neu sau khi bo van con du so luong yeu cau thi dung ban da loc do (khach san
    co diem se khong bi lan lon voi khach san thieu diem); neu khong du thi quay lai
    dung toan bo tap lien quan, sap theo diem cao->thap (khach san thieu diem xep cuoi).
    """
    scored_only = df[df[C_SCORE].notna()]
    pool = scored_only if len(scored_only) >= nums else df
    return pool.sort_values(C_SCORE, ascending=False, na_position="last").head(nums)


def search_cosine(df_hotel: pd.DataFrame, art: dict, search_str: str, nums=5) -> pd.DataFrame:
    """
    Khop search_cosine trong notebook — TfidfVectorizer + cosine_similarity.
    Trong tap khach san LIEN QUAN ve noi dung (similarity > 0), uu tien hien thi
    khach san co Total_Score cao truoc (xem rank_by_score_with_fallback).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    words = preprocess_text(search_str, art.get("text_res", {}))
    if not words or art["vectorizer"] is None or art["tfidf_matrix"] is None:
        return pd.DataFrame()

    search_vec = art["vectorizer"].transform([" ".join(words)])
    sim_cos = cosine_similarity(search_vec, art["tfidf_matrix"]).flatten()

    relevant_idx = np.flatnonzero(sim_cos > 0)
    if len(relevant_idx) == 0:
        return pd.DataFrame()

    pool = df_hotel.iloc[relevant_idx].copy()
    pool["similarity"] = sim_cos[relevant_idx]
    return rank_by_score_with_fallback(pool, nums)


# ---- Yeu cau 2: Collaborative filtering (khop Project1_Request2.ipynb) ----
def get_knn_similar_hotels(df_hotel: pd.DataFrame, cf_art: dict, hotel_id, nums=5) -> pd.DataFrame:
    """
    Khop buoc 'Thu tim Hotel gan nhat' trong notebook — NearestNeighbors(metric='cosine')
    tren ma tran Hotel x User. Tra ve Top-N khach san gan nhat theo khoang cach cosine,
    da loai bo chinh no.

    Luu y: do moi user chi danh gia 1 khach san (xem canh bao o UI), khoang cach cosine
    giua MOI CAP khach san deu bang nhau (hoa tuyet doi). Neu giu nguyen thu tu tra ve tu
    sklearn, ket qua se luon la CUNG MOT bo khach san co dinh du ban chon khach san nao
    de tim (vi sklearn phai bo tie bang thu tu chi so co dinh, khong phu thuoc query).
    De tranh cam giac "doi khach san ma khong doi ket qua", minh xao lai thu tu cac
    candidate dang hoa diem bang 1 hash on dinh theo hotel_id truy van — ket qua se khac
    nhau tuy khach san ban chon, dung nhu ky vong UX, dua tren "hoa" giong nhau ve ban chat.
    """
    import hashlib

    knn, matrix, hotel_index = cf_art.get("knn"), cf_art.get("matrix"), cf_art.get("hotel_index")
    if knn is None or matrix is None or hotel_index is None:
        return pd.DataFrame()

    match = hotel_index[hotel_index[C_ID] == hotel_id]
    if match.empty:
        return pd.DataFrame()
    pos = int(match["Hotel_Index"].iloc[0])

    # Lay du ung vien de co the xao lai thu tu trong cac nhom hoa diem
    n_query = min(max(nums * 5, nums + 1), matrix.shape[0])
    distances, indices = knn.kneighbors(matrix[pos], n_neighbors=n_query)

    pairs = [(int(i), float(d)) for i, d in zip(indices.flatten(), distances.flatten()) if i != pos]
    if not pairs:
        return pd.DataFrame()

    def tie_break_key(pair):
        idx, dist = pair
        # On dinh theo (hotel truy van, ung vien) — cung 1 khach san luon ra cung 1 thu tu,
        # nhung doi khach san truy van se ra thu tu khac.
        h = hashlib.md5(f"{hotel_id}-{idx}".encode("utf-8")).hexdigest()
        return (round(dist, 6), h)

    pairs.sort(key=tie_break_key)
    pairs = pairs[:nums]

    idx_to_id = hotel_index.set_index("Hotel_Index")[C_ID]
    ids = [idx_to_id.loc[p[0]] for p in pairs]
    dist_map = dict(zip(ids, [p[1] for p in pairs]))

    out = df_hotel[df_hotel[C_ID].isin(ids)].copy()
    out["similarity"] = out[C_ID].map(lambda h: 1 - dist_map[h])
    return out.sort_values("similarity", ascending=False)


# ---- Yeu cau 3: Insight (khop tung ham trong Project1_Request3.ipynb) ----
def get_hotel_overview(df_hotel: pd.DataFrame, hotel_id) -> dict | None:
    """Khop get_hotel_overview — thong tin tong quan: ten, hang sao, dia chi, diem TB."""
    row = df_hotel[df_hotel[C_ID] == hotel_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "Hotel_Name": row.get(C_NAME),
        "Hotel_Rank": row.get(C_RANK),
        "Hotel_Address": row.get(C_ADDR),
        "Total_Score": row.get(C_SCORE),
        "comments_count": row.get(C_COUNT, 0),
    }


def analyze_strengths_weaknesses(df_hotel: pd.DataFrame, hotel_id) -> pd.Series | None:
    """Khop analyze_strengths_weaknesses — sap xep 6 tieu chi diem tu manh nhat den yeu nhat."""
    row = df_hotel[df_hotel[C_ID] == hotel_id]
    if row.empty:
        return None
    row = row.iloc[0]

    cols = [c for c in SCORE_COLS if c in row.index]
    scores = row[cols].dropna()
    scores = pd.to_numeric(
        scores.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    ).dropna()
    return scores.sort_values(ascending=False) if not scores.empty else None


def analyze_customer_stats(df_cmt: pd.DataFrame, hotel_id) -> dict | None:
    """Khop analyze_customer_stats — quoc tich, hinh thuc di du lich, xu huong theo thoi gian."""
    if df_cmt.empty or CM_ID not in df_cmt.columns:
        return None
    reviews = df_cmt[df_cmt[CM_ID] == hotel_id]
    if reviews.empty:
        return None

    nationality_counts = reviews[CM_NATION].value_counts().head(10) if CM_NATION in reviews else pd.Series(dtype=int)
    group_counts = reviews[CM_GROUP].value_counts() if CM_GROUP in reviews else pd.Series(dtype=int)
    trend = (reviews.dropna(subset=[CM_YEARMONTH]).groupby(CM_YEARMONTH).size()
             if CM_YEARMONTH in reviews else pd.Series(dtype=int))

    return {"nationality": nationality_counts, "group": group_counts, "trend": trend, "n_reviews": len(reviews)}


def analyze_keywords(df_cmt: pd.DataFrame, hotel_id, top_n=15) -> dict | None:
    """Khop analyze_keywords — tu khoa noi bat trong danh gia tich cuc (>=8) va tieu cuc (<6)."""
    if df_cmt.empty or CM_ID not in df_cmt.columns:
        return None
    reviews = df_cmt[df_cmt[CM_ID] == hotel_id]
    if reviews.empty:
        return None

    positive_reviews = reviews[reviews[CM_SCORE] >= POSITIVE_THRESHOLD][CM_BODY].dropna()
    negative_reviews = reviews[reviews[CM_SCORE] < NEGATIVE_THRESHOLD][CM_BODY].dropna()

    pos_tokens = [tok for body in positive_reviews for tok in tokenize_vi(body)]
    neg_tokens = [tok for body in negative_reviews for tok in tokenize_vi(body)]

    return {
        "positive_top": Counter(pos_tokens).most_common(top_n),
        "negative_top": Counter(neg_tokens).most_common(top_n),
        "n_positive": len(positive_reviews),
        "n_negative": len(negative_reviews),
    }


def compare_with_system(df_hotel: pd.DataFrame, hotel_id, system_avg: pd.Series) -> pd.DataFrame | None:
    """Khop compare_with_system — so sanh tung tieu chi voi trung binh toan he thong."""
    row = df_hotel[df_hotel[C_ID] == hotel_id]
    if row.empty:
        return None
    row = row.iloc[0]

    cols = [c for c in system_avg.index if c in row.index]
    comparison = pd.DataFrame({
        "Khách sạn": pd.to_numeric(row[cols], errors="coerce"),
        "Trung bình hệ thống": system_avg[cols],
    })
    comparison["Chênh lệch"] = comparison["Khách sạn"] - comparison["Trung bình hệ thống"]
    return comparison


def generate_report(df_hotel: pd.DataFrame, df_cmt: pd.DataFrame, hotel_id, system_avg: pd.Series) -> dict | None:
    """Khop generate_report — gom toan bo phan tich + rut ra insight tu dong."""
    overview = get_hotel_overview(df_hotel, hotel_id)
    if overview is None:
        return None

    scores = analyze_strengths_weaknesses(df_hotel, hotel_id)
    cust_stats = analyze_customer_stats(df_cmt, hotel_id)
    keywords = analyze_keywords(df_cmt, hotel_id)
    comparison = compare_with_system(df_hotel, hotel_id, system_avg)

    insights = []
    if scores is not None and len(scores) >= 2:
        insights.append(
            f"Tiêu chí điểm cao nhất: **{scores.index[0]}** ({scores.iloc[0]:.1f}/10). "
            f"Tiêu chí điểm thấp nhất: **{scores.index[-1]}** ({scores.iloc[-1]:.1f}/10)."
        )
    if comparison is not None:
        below_avg = comparison[comparison["Chênh lệch"] < 0]
        if not below_avg.empty:
            worst_gap = below_avg["Chênh lệch"].idxmin()
            insights.append(
                f"Tiêu chí **{worst_gap}** thấp hơn trung bình hệ thống "
                f"**{abs(below_avg.loc[worst_gap, 'Chênh lệch']):.2f} điểm**."
            )
        else:
            insights.append("Khách sạn có điểm số cao hơn hoặc bằng trung bình hệ thống ở tất cả các tiêu chí.")
    if cust_stats is not None and not cust_stats["nationality"].empty:
        top_nat = cust_stats["nationality"].index[0]
        top_nat_count = int(cust_stats["nationality"].iloc[0])
        insights.append(
            f"Quốc tịch chiếm tỉ trọng lớn nhất: **{top_nat}** ({top_nat_count} lượt đánh giá)."
        )
    if keywords is not None and keywords["negative_top"]:
        top_neg_word, top_neg_count = keywords["negative_top"][0]
        insights.append(
            f"Từ khoá xuất hiện nhiều nhất trong đánh giá tiêu cực: **'{top_neg_word}'** ({top_neg_count} lần)."
        )

    return {
        "overview": overview, "scores": scores, "customer_stats": cust_stats,
        "keywords": keywords, "comparison": comparison, "insights": insights,
    }


# =========================================================================
# TANG 7 — UI
# =========================================================================
HERO_BG_PATH = os.path.join(ASSETS_DIR, "beach_hero.jpg")


@st.cache_data
def load_hero_bg_b64() -> str:
    """Doc anh nen hero, encode base64 de nhung thang vao CSS (khong can cau hinh static server)."""
    import base64
    if not os.path.exists(HERO_BG_PATH):
        return ""
    with open(HERO_BG_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


_bg_b64 = load_hero_bg_b64()
if _bg_b64:
    _hero_bg_css = (
        f"background-image: linear-gradient(180deg, rgba(6,34,46,0.18) 0%, rgba(6,34,46,0.42) 100%), "
        f"url('data:image/jpeg;base64,{_bg_b64}'); "
        f"background-size: cover; background-position: center;"
    )
else:
    # Du phong neu chua co assets/beach_hero.jpg — giu gradient bien lam nen
    _hero_bg_css = "background: linear-gradient(135deg, var(--deepsea) 0%, var(--teal) 48%, var(--turquoise) 100%);"

HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Public+Sans:wght@400;500;600&display=swap');

:root {
    --deepsea: #073B4C;
    --teal: #0E7C86;
    --turquoise: #2EC4B6;
    --sand: #F4E3C1;
    --coral: #FF6F59;
    --cream: #FDFBF7;
    --agoda-blue: #1D6FE0;
    --agoda-blue-dark: #164FA3;
}

/* Them khoang dem TREN CUNG rong rai, chu dinh ro rang (khong dua vao mac dinh Streamlit
   mo ho) — de dam bao KHONG BAO GIO bi thanh cong cu noi cua Streamlit Cloud
   (Fork/GitHub/Deploy) de len tren banner/tab bar, du cuon hay resize the nao. */
.block-container { padding-top: 3.5rem !important; }

.hero-wrap {
    position: relative;
    margin: 0 -1rem 1.75rem -1rem;
    padding: 3.2rem 2.2rem 3.6rem 2.2rem;
    __HERO_BG__
    overflow: hidden;
    text-align: center;
    border-radius: 0 0 28px 28px;
}
.hero-eyebrow {
    display: inline-block;
    font-family: 'Public Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--deepsea);
    background: var(--sand);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
    position: relative;
    z-index: 2;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-weight: 600;
    font-size: clamp(1.8rem, 3.6vw, 2.7rem);
    line-height: 1.22;
    color: var(--cream) !important;
    text-shadow: 0 2px 18px rgba(6,34,46,0.55);
    margin: 0 auto 0.6rem auto !important;
    position: relative;
    z-index: 2;
    max-width: 42rem;
}
.hero-sub {
    font-family: 'Public Sans', sans-serif;
    font-size: 1.05rem;
    color: rgba(253, 251, 247, 0.95) !important;
    text-shadow: 0 1px 12px rgba(6,34,46,0.55);
    max-width: 38rem;
    margin-left: auto !important;
    margin-right: auto !important;
    position: relative;
    z-index: 2;
}
/* Da bo hoan toan .hero-wave (SVG song trang tri o day banner) — day la phan tu
   position:absolute duy nhat nam sat tab bar, nghi ngo la nguyen nhan gay che chu
   tren mot so trinh duyet/thiet bi ma khong the tai hien de test cuc bo. */

/* Khoi tim kiem thu gon kieu Agoda: the noi, bo, can giua, khong chiem full width */
.st-key-search_card, .st-key-cf_card, .st-key-insight_card {
    background: var(--cream);
    border-radius: 18px;
    box-shadow: 0 18px 40px -18px rgba(6,34,46,0.35);
    padding: 1.5rem 1.7rem 1.2rem 1.7rem;
    max-width: 760px;
    margin: 1.2rem auto 1.6rem auto;
    position: relative;
    z-index: 5;
}
@media (max-width: 820px) {
    .st-key-search_card, .st-key-cf_card, .st-key-insight_card { margin-top: 0.5rem; max-width: 100%; }
}

/* Can giua thanh tab (Yeu cau 1 / Yeu cau 3) */
[data-testid="stTabs"] [role="tablist"] {
    justify-content: center;
}

/* Nut chinh: bo tron, mau xanh, can giua, khong keo full-width */
div[data-testid="stElementContainer"]:has(div[data-testid="stFormSubmitButton"]),
div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) {
    width: 100% !important;
}
div[data-testid="stFormSubmitButton"], div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
div[data-testid="stFormSubmitButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
div[data-testid="stButton"] button[kind="primary"] {
    border-radius: 999px !important;
    padding: 0.6rem 2.8rem !important;
    font-weight: 600 !important;
    background-color: var(--agoda-blue) !important;
    border-color: var(--agoda-blue) !important;
    color: #FFFFFF !important;
    width: auto !important;
}
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: var(--agoda-blue-dark) !important;
    border-color: var(--agoda-blue-dark) !important;
}

/* The ket qua khach san: gon hon, anh dai dien, ten to & dam ro rang */
.hotel-thumb svg {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 12px;
}
.hotel-thumb img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    display: block;
    border-radius: 12px;
}
.hotel-detail-thumb {
    text-align: center;
}
.hotel-detail-thumb svg, .hotel-detail-thumb img {
    width: 50% !important;
    max-width: 50% !important;
    aspect-ratio: 1 / 1 !important;
    object-fit: cover !important;
    display: inline-block !important;
    border-radius: 12px !important;
    margin-bottom: 0.8rem !important;
}
@media (max-width: 700px) {
    .hotel-detail-thumb svg, .hotel-detail-thumb img { width: 80% !important; max-width: 80% !important; }
}
.hotel-detail-meta {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--deepsea) !important;
    margin-bottom: 1rem;
}
.hotel-detail-meta .meta-star {
    font-size: 1.35rem;
}
.hotel-detail-meta .meta-pin {
    font-size: 1.15rem;
}
.meta-map-link {
    color: var(--agoda-blue) !important;
    text-decoration: none !important;
}
.meta-map-link:hover {
    text-decoration: underline !important;
}
.overview-label {
    font-weight: 700 !important;
    color: var(--deepsea) !important;
    font-size: 0.95rem;
    margin-bottom: 0.15rem;
}
.overview-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #262730;
}
.overview-address {
    font-size: 0.95rem;
    font-weight: 600;
    color: #262730;
    line-height: 1.35;
}
.hotel-card-meta {
    font-size: 0.88rem !important;
    margin-bottom: 0.2rem !important;
}
.hotel-card-meta .meta-star {
    font-size: 1rem;
}
.hotel-card-meta .meta-pin {
    font-size: 0.92rem;
}
.hotel-card-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.2rem;
    line-height: 1.28;
    color: var(--deepsea);
    margin-bottom: 0.1rem;
}
[class*="st-key-result_"] {
    padding: 0.5rem 0.9rem !important;
}
</style>

<div class="hero-wrap">
    <span class="hero-eyebrow">Nha Trang · Khánh Hoà</span>
    <h1 class="hero-title">Hệ thống gợi ý tìm kiếm khách sạn ở Nha Trang</h1>
    <p class="hero-sub">
        Đồ án tốt nghiệp Data Science · Nhóm thực hiện: Kim Thanh – Quang Lợi
    </p>
</div>
""".replace("__HERO_BG__", _hero_bg_css)
st.markdown(HERO_HTML, unsafe_allow_html=True)

if not _bg_b64:
    st.caption(
        "💡 Chưa thấy ảnh nền `assets/beach_hero.jpg` — đang tạm dùng nền gradient. "
        "Copy file ảnh vào thư mục `assets/` cạnh `app.py` để hiện ảnh biển thật."
    )

if not os.path.exists(HOTEL_PATH):
    st.error("Chưa có dữ liệu để chạy. Sinh dữ liệu và mô hình rồi tải lại trang.")
    st.code("python build_hotel_artifacts.py", language="bash")
    st.stop()

df_hotel = load_hotels()
df_cmt = load_comments()
content_art = load_content_model()
cf_art = load_cf_model()
hotel_photos = load_hotel_photos()
system_avg = compute_system_avg(df_hotel)

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 1. Gợi ý theo nội dung",
        "👥 2. Gợi ý theo lọc cộng tác",
        "📊 3. Insight cho chủ khách sạn",
    ],
    key="active_tab_label",
    on_change="rerun",
)

# ------------------------------------------------------------------ TAB 1
with tab1:
    if "tab1_viewing_id" in st.session_state:
        render_hotel_detail_page(st.session_state["tab1_viewing_id"], df_hotel, df_cmt,
                                  hotel_photos, state_key="tab1_viewing_id")
    else:
        st.markdown(
            '<p style="text-align:center; color:#6b7280; font-size:0.875rem;">'
            'Mô tả điều bạn mong muốn để hệ thống tìm khách sạn phù hợp.</p>',
            unsafe_allow_html=True,
        )

        if content_art["cosine_sim"] is None and content_art["vectorizer"] is None:
            st.error(
                "Chưa có mô hình content-based. Chạy `python build_hotel_artifacts.py` để tạo "
                "`cosine_sim.npy`."
            )
        else:
            with st.container(key="search_card"):
                with st.form("search_form"):
                    query, top_n = render_search_inputs()
                    submitted = st.form_submit_button("🔍 Tìm kiếm", type="primary")

            if submitted:
                res = search_cosine(df_hotel, content_art, query or "", nums=top_n)
                st.session_state["tab1_results"] = res

            if "tab1_results" in st.session_state:
                st.divider()
                res = st.session_state["tab1_results"]
                st.subheader(f"Top {len(res)} khách sạn phù hợp nhất")
                render_results(res, hotel_photos, state_key="tab1_viewing_id")

# ------------------------------------------------------------------ TAB 2
with tab2:
    if "tab2_viewing_id" in st.session_state:
        render_hotel_detail_page(st.session_state["tab2_viewing_id"], df_hotel, df_cmt,
                                  hotel_photos, state_key="tab2_viewing_id")
    elif cf_art.get("knn") is None or cf_art.get("hotel_index") is None:
        st.error(
            "Chưa có mô hình Collaborative filtering. Chạy `python build_hotel_artifacts.py` để tạo "
            "`cf_knn_model.joblib`."
        )
    else:
        cf_hotel_ids = set(cf_art["hotel_index"][C_ID])
        cf_hotel_names = df_hotel.loc[df_hotel[C_ID].isin(cf_hotel_ids), C_NAME].tolist()

        if not cf_hotel_names:
            st.warning("Không có khách sạn nào đủ dữ liệu đánh giá để dùng Collaborative filtering.")
        else:
            with st.container(key="cf_card"):
                with st.form("cf_form"):
                    picked_name = st.selectbox(
                        "Khách sạn bạn đang quan tâm",
                        cf_hotel_names,
                        key="cf_hotel",
                        help="Chỉ hiển thị các khách sạn đã có ít nhất 1 đánh giá trong dữ liệu.",
                    )
                    cf_top_n = st.slider("Số khách sạn gợi ý", 3, 20, TOP_N_DEFAULT, key="cf_top_n")
                    cf_submitted = st.form_submit_button("👥 Tìm kiếm", type="primary")

            if cf_submitted:
                picked_id = df_hotel.loc[df_hotel[C_NAME] == picked_name, C_ID].iloc[0]
                res_cf = get_knn_similar_hotels(df_hotel, cf_art, picked_id, nums=cf_top_n)
                st.session_state["tab2_results"] = res_cf

            if "tab2_results" in st.session_state:
                st.divider()
                res_cf = st.session_state["tab2_results"]
                st.subheader(f"Top {len(res_cf)} khách sạn tương tự (Item-Based KNN)")
                render_results(res_cf, hotel_photos, state_key="tab2_viewing_id")

# ------------------------------------------------------------------ TAB 3
with tab3:

    if df_cmt.empty:
        st.warning(
            "Chưa có dữ liệu đánh giá (`hotel_comments`). Một số phần (thống kê khách hàng, "
            "từ khoá, xu hướng) sẽ không hiển thị được cho tới khi có dữ liệu này."
        )

    with st.container(key="insight_card"):
        hotel_id3 = render_hotel_picker(df_hotel, key="insight_hotel")
        xem_bao_cao = st.button("📊 Xem báo cáo insight", type="primary")

    if xem_bao_cao:
        rep = generate_report(df_hotel, df_cmt, hotel_id3, system_avg)
        if rep is None:
            st.error("Không tìm thấy khách sạn này.")
        else:
            ov = rep["overview"]
            st.divider()
            st.subheader(f"1. Tổng quan — {ov['Hotel_Name']}")

            def _overview_stat(icon, label, value):
                st.markdown(
                    f'<div class="overview-label">{icon} {label}</div>'
                    f'<div class="overview-value">{value}</div>',
                    unsafe_allow_html=True,
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                _overview_stat("⭐", "Hạng sao", ov["Hotel_Rank"])
            with c2:
                _overview_stat("🎯", "Tổng điểm TB", f"{fmt_score(ov['Total_Score'])}/10")
            with c3:
                _overview_stat("💬", "Số lượt đánh giá", f"{int(ov['comments_count']):,}".replace(",", "."))
            with c4:
                st.markdown(
                    '<div class="overview-label">📍 Địa chỉ</div>'
                    f'<div class="overview-address">{ov["Hotel_Address"]}</div>',
                    unsafe_allow_html=True,
                )

            st.subheader("2. Điểm mạnh & điểm yếu")
            scores = rep["scores"]
            if scores is None or scores.empty:
                st.info("Khách sạn chưa có đủ dữ liệu điểm chi tiết.")
            else:
                colors = ["#2ecc71" if v == scores.max() else "#e74c3c" if v == scores.min() else "#3498db"
                          for v in scores.values]
                # Chieu cao ty le theo so tieu chi thuc co — tranh 1 thanh duy nhat bi keo
                # gian day het chieu cao co dinh khi khach san chi co du lieu 1-2 tieu chi.
                fig_height = max(1.12, 0.385 * len(scores) + 0.63)
                fig, ax = plt.subplots(figsize=(5.6, fig_height))
                ax.barh(scores.index, scores.values, color=colors, height=0.55)
                ax.set_xlabel("Điểm (/10)")
                ax.set_xlim(0, 10)
                for i, v in enumerate(scores.values):
                    ax.text(v + 0.1, i, f"{v:.1f}", va="center")
                plt.tight_layout()
                st.pyplot(fig, width=780)
                plt.close(fig)
                st.caption(
                    f"Mạnh nhất: **{scores.index[0]}** ({scores.iloc[0]:.1f}) · "
                    f"Yếu nhất: **{scores.index[-1]}** ({scores.iloc[-1]:.1f})"
                )

            st.subheader("3. Thống kê khách hàng")
            cust = rep["customer_stats"]
            if cust is None:
                st.info("Khách sạn chưa có đánh giá nào.")
            else:
                c1, c2 = st.columns([1, 1.4])
                with c1:
                    nat = cust["nationality"].astype(float)
                    nat = nat[nat > 0].sort_values(ascending=False)
                    if nat.sum() > 0:
                        # Gop cac lat < MIN_PIE_PCT% vao mot nhom "Khac" cho de nhin
                        total_nat = nat.sum()
                        share = nat / total_nat * 100
                        major = nat[share >= MIN_PIE_PCT]
                        minor = nat[share < MIN_PIE_PCT]
                        if len(minor) > 0:
                            other_label = f"Khác ({len(minor)} quốc tịch)"
                            major = pd.concat(
                                [major, pd.Series({other_label: minor.sum()})]
                            )
                        fig, ax = plt.subplots(figsize=(3.6, 4.0))
                        colors = plt.get_cmap("tab20").colors[: len(major)]
                        wedges, _, autotexts = ax.pie(
                            major.values,
                            autopct="%1.0f%%",
                            startangle=90,
                            counterclock=False,
                            colors=colors,
                            pctdistance=0.72,
                            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
                        )
                        for t in autotexts:
                            t.set_fontsize(8.5)
                            t.set_color("white")
                            t.set_fontweight("bold")
                        ax.set_title("Top quốc tịch khách hàng", fontsize=11)
                        ax.legend(
                            wedges,
                            [f"{k} · {v / total_nat * 100:.0f}%" for k, v in major.items()],
                            loc="upper center",
                            bbox_to_anchor=(0.5, 0.02),
                            ncol=2,
                            fontsize=7.5,
                            frameon=False,
                        )
                        plt.tight_layout()
                        st.pyplot(fig, width=350)
                        plt.close(fig)
                with c2:
                    if not cust["group"].empty:
                        grp = cust["group"].sort_values(ascending=False)
                        pos = np.arange(len(grp))
                        labels = ["\n".join(textwrap.wrap(str(s), 11)) for s in grp.index]
                        fig, ax = plt.subplots(figsize=(5.6, 4.0))
                        ax.bar(pos, grp.values, color="#3498db", width=0.55)
                        ax.set_title("Hình thức đi du lịch", fontsize=11)
                        ax.set_ylabel("Số lượt đánh giá")
                        ax.set_xlabel("")
                        ax.set_xticks(pos)
                        ax.set_xticklabels(labels, fontsize=8)
                        # Keo dai truc Ox de cac nhan khong bi dinh nhau
                        ax.set_xlim(-0.7, len(grp) - 0.3)
                        ax.set_ylim(0, float(grp.max()) * 1.18)
                        for x, v in zip(pos, grp.values):
                            ax.text(x, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
                        ax.spines[["top", "right"]].set_visible(False)
                        ax.grid(axis="y", linestyle=":", alpha=0.4)
                        ax.set_axisbelow(True)
                        plt.tight_layout()
                        st.pyplot(fig, width=520)
                        plt.close(fig)
                if len(cust["trend"]) > 1:
                    fig, ax = plt.subplots(figsize=(7, 2.45))
                    cust["trend"].plot(ax=ax, marker="o", color="#e67e22")
                    ax.set_title("Xu hướng số lượt đánh giá theo thời gian")
                    ax.set_xlabel("Tháng")
                    ax.set_ylabel("Số lượt đánh giá")
                    plt.tight_layout()
                    st.pyplot(fig, width=780)
                    plt.close(fig)

            st.subheader("4. Từ khoá nổi bật")
            kw = rep["keywords"]
            if kw is None:
                st.info("Khách sạn chưa có đánh giá nào.")
            else:
                st.caption(f"{kw['n_positive']} đánh giá tích cực (≥{POSITIVE_THRESHOLD}) · "
                           f"{kw['n_negative']} đánh giá tiêu cực (<{NEGATIVE_THRESHOLD})")
                c1, c2 = st.columns(2)
                with c1:
                    if kw["positive_top"]:
                        words, counts = zip(*kw["positive_top"])
                        fig, ax = plt.subplots(figsize=(4.2, 3.15))
                        ax.barh(words[::-1], counts[::-1], color="#2ecc71")
                        ax.set_title("Top từ khoá — Tích cực")
                        plt.tight_layout()
                        st.pyplot(fig, width=390)
                        plt.close(fig)
                    else:
                        st.caption("Không đủ dữ liệu đánh giá tích cực.")
                with c2:
                    if kw["negative_top"]:
                        words_n, counts_n = zip(*kw["negative_top"])
                        fig, ax = plt.subplots(figsize=(4.2, 3.15))
                        ax.barh(words_n[::-1], counts_n[::-1], color="#e74c3c")
                        ax.set_title("Top từ khoá — Tiêu cực")
                        plt.tight_layout()
                        st.pyplot(fig, width=390)
                        plt.close(fig)
                    else:
                        st.caption("Không đủ dữ liệu đánh giá tiêu cực để phân tích.")

            st.subheader("5. So sánh với trung bình hệ thống")
            comparison = rep["comparison"]
            if comparison is None or comparison.empty:
                st.info("Không đủ dữ liệu để so sánh.")
            else:
                fig, ax = plt.subplots(figsize=(6.3, 3.15))
                x = np.arange(len(comparison))
                width = 0.35
                ax.bar(x - width / 2, comparison["Khách sạn"], width, label="Khách sạn", color="#3498db")
                ax.bar(x + width / 2, comparison["Trung bình hệ thống"], width,
                       label="Trung bình hệ thống", color="#95a5a6")
                ax.set_xticks(x)
                ax.set_xticklabels(comparison.index, rotation=30, ha="right")
                ax.set_ylabel("Điểm (/10)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig, width=780)
                plt.close(fig)

                # Bieu do ngang rieng cho Chenh lech — de hinh dung cao/thap va am/duong
                diffs = comparison["Chênh lệch"].dropna()
                if diffs.empty:
                    st.caption("Không đủ dữ liệu để vẽ biểu đồ chênh lệch.")
                else:
                    fig2, ax2 = plt.subplots(figsize=(6.3, max(1.26, 0.35 * len(diffs) + 0.56)))
                    colors2 = ["#0E7C86" if v >= 0 else "#E74C3C" for v in diffs]
                    ax2.barh(diffs.index, diffs.values, color=colors2)
                    ax2.axvline(0, color="#073B4C", linewidth=0.9)
                    ax2.set_xlabel("Chênh lệch so với trung bình hệ thống (điểm)")
                    for i, v in enumerate(diffs.values):
                        ax2.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                                  va="center", ha="left" if v >= 0 else "right", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2, width=780)
                    plt.close(fig2)

                render_comparison_table(comparison)

            st.subheader("6. Một số insight nổi bật")
            if rep["insights"]:
                for line in rep["insights"]:
                    st.markdown(f"- {line}")
            else:
                st.info("Chưa đủ dữ liệu để rút ra insight.")

# -------------------------------------------------------- LUON HIEN DUOI 3 TAB
# Dai goi y nhanh nam NGOAI cac tab — luon hien du dang o tab nao hay da tim kiem
# hay chua, cho trang sinh dong hon thay vi bien mat sau khi bam tim.
st.divider()
render_quick_suggestions()
