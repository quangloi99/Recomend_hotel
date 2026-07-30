# -*- coding: utf-8 -*-
"""
Sinh data/ va models/ cho app_hotel.py — dung DUNG pipeline trong Project1_Request1.ipynb.

    python build_hotel_artifacts.py
    python build_hotel_artifacts.py --info ... --cmt ... --stopwords ... --envn ...

Pipeline (khop notebook):
    fillna -> loai mo ta loi -> clean_raw_text -> pyvi tokenize
    -> clean_content_column (xoa so, bo ky tu dac biet, bo stopword, dich Anh->Viet)
    -> Gensim (filter_extremes no_below=5 no_above=0.5) + Cosine (TfidfVectorizer analyzer='word')
"""
import argparse
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================================
#  SUA 4 DONG NAY 1 LAN — sau do khong can nhap gi nua
#  (de nguyen ten file khong duong dan neu ban copy file vao cung thu muc)
# =====================================================================
HOTEL_INFO_CSV = r"C:\LOI\DL_307\KHOA_7\Data_Agoda_raw-20260714T231048Z-1-001\Data_Agoda_raw\hotel_info.csv"
HOTEL_COMMENTS_CSV = r"C:\LOI\DL_307\KHOA_7\Data_Agoda_raw-20260714T231048Z-1-001\Data_Agoda_raw\hotel_comments.csv"
STOPWORDS_TXT = r"C:\LOI\DL_307\KHOA_7\drive-download-20260714T231345Z-1-001\files\vietnamese-stopwords.txt"
EN_VN_TXT = r"C:\LOI\DL_307\KHOA_7\files-20260714T231050Z-1-001\files\english-vnmese.txt"


def resolve(path, filename):
    """Uu tien duong dan cau hinh o tren; neu khong co thi tim canh script."""
    if path and os.path.exists(path):
        return path
    return os.path.join(BASE_DIR, filename)

ERROR_VALUES = ["#NAME?", "#REF!", "#VALUE!", "#N/A", ""]

SPECIAL_CHARS = ["", " ", ",", ".", "...", "-", ":", ";", "?", "%", "(", ")",
                 "+", "/", "'", "&", '"', "!", "\u201c", "\u201d", "\u2013", "\u2026"]

EXTRA_STOPWORDS = [
    "khách_sạn", "du_khách", "quý_khách", "khách_hàng", "dịch_vụ",
    "tuyệt_vời", "lý_tưởng", "tận_hưởng", "thưởng_thức", "mang_lại",
    "đáp_ứng", "lựa_chọn", "trải", "nghiệm",
    "không", "nằm", "giúp", "nơi", "sự", "việc",
]

ENGLISH_STOPWORDS = set([
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "these", "those",
    "and", "or", "but", "in", "on", "at", "to", "of", "for", "with", "from", "by",
    "you", "your", "we", "our", "it", "its", "can", "will", "be", "has", "have",
    "as", "all", "so", "if", "not", "no", "do", "does",
])


def log(m):
    print(m, flush=True)


# ============================ TIEN XU LY (khop notebook) ============================
def clean_raw_text(row):
    """Bo khoang trang thua + bo ten khach san khoi mo ta."""
    text = re.sub(r"\s+", " ", str(row["Hotel_Description"])).strip()
    name = re.sub(r"\(.*?\)", "", str(row["Hotel_Name"])).strip()
    if len(name) > 3:
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def make_translate(en_vn_dict):
    def translate_english_tokens(tokens):
        out = []
        for t in tokens:
            tl = t.lower()
            if tl in ENGLISH_STOPWORDS:
                continue
            out.append(en_vn_dict.get(tl, t))
        return out
    return translate_english_tokens


def clean_tokens(tokens, all_stopwords, translate_fn):
    """Logic trong clean_content_column, ap dung cho 1 document."""
    tokens = [re.sub(r"[0-9]+", "", t) for t in tokens]
    tokens = [re.sub(r"[^\w_]", "", t.lower()) for t in tokens if t not in SPECIAL_CHARS]
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 1]
    return translate_fn(tokens)


# ---- Khop Project1_Request3.ipynb (Insight) ----
SCORE_COLS = ["Total_Score", "Location", "Cleanliness", "Service",
              "Facilities", "Value_for_money", "Comfort_and_room_quality"]


def convert_comma_number(x):
    """Chuyen chuoi so kieu '8,8' -> float 8.8. Tra ve NaN neu khong hop le."""
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return np.nan


def extract_star_rank(x):
    """Tach so sao tu chuoi '5 sao trên 5' -> 5.0. 'No information' -> NaN."""
    if pd.isna(x):
        return np.nan
    match = re.search(r"([\d\.]+)\s*sao", str(x))
    return float(match.group(1)) if match else np.nan


def parse_review_date(x):
    """Parse 'Đã nhận xét vào 30 tháng 7 2023' -> Timestamp(2023-07-30)."""
    if pd.isna(x):
        return pd.NaT
    match = re.search(r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*(\d{4})", str(x))
    if not match:
        return pd.NaT
    day, month, year = match.groups()
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except ValueError:
        return pd.NaT


# ============================ 1. DU LIEU ============================
def build_data(info_path, cmt_path, sw_path, envn_path):
    from pyvi.ViTokenizer import tokenize

    if not os.path.exists(info_path):
        sys.exit(f"[LOI] Khong thay {info_path}")

    df = pd.read_csv(info_path)
    log(f"[data] hotel_info: {df.shape}")
    for c in ("Hotel_ID", "Hotel_Name", "Hotel_Description"):
        if c not in df.columns:
            sys.exit(f"[LOI] hotel_info thieu cot {c}. Cac cot: {list(df.columns)}")

    df["Hotel_Description"] = df["Hotel_Description"].fillna("")
    df1 = df[~df["Hotel_Description"].str.strip().isin(ERROR_VALUES)].reset_index(drop=True)
    log(f"[data] loai {len(df) - len(df1)} khach san mo ta loi/rong -> con {len(df1)}")

    # Ep cac cot so ve float — CSV Viet Nam thuong dung dau phay thap phan (vd "8,8")
    for col in SCORE_COLS:
        if col in df1.columns:
            df1[col] = pd.to_numeric(
                df1[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )
            n_bad = df1[col].isna().sum()
            if n_bad:
                log(f"[canh bao] cot {col}: {n_bad} gia tri khong doi duoc sang so")

    # stopword
    if os.path.exists(sw_path):
        with open(sw_path, "r", encoding="utf-8") as f:
            stop_words = f.read().split("\n")
    else:
        stop_words = []
        log(f"[canh bao] khong thay {sw_path} -> chi dung {len(EXTRA_STOPWORDS)} stopword bo sung")
    all_stopwords = set(stop_words) | set(EXTRA_STOPWORDS)
    log(f"[data] tong stopword: {len(all_stopwords)}")

    # tu dien Anh -> Viet
    en_vn_dict = {}
    if os.path.exists(envn_path):
        with open(envn_path, "r", encoding="utf-8") as f:
            en_vn_dict = dict(l.strip().split("\t") for l in f if "\t" in l)
        log(f"[data] tu dien Anh-Viet: {len(en_vn_dict)} tu")
    else:
        log(f"[canh bao] khong thay {envn_path} -> bo qua buoc dich Anh->Viet")
    translate_fn = make_translate(en_vn_dict)

    # Content -> Content_wt -> Content_clean
    df1["Content"] = df1.apply(clean_raw_text, axis=1)
    log("[data] dang tach tu bang pyvi (mat vai chuc giay)...")
    df1["Content_wt"] = df1["Content"].apply(tokenize)

    content_tokens = [clean_tokens(x.split(), all_stopwords, translate_fn) for x in df1["Content_wt"]]
    df1["Content_clean"] = [" ".join(t) for t in content_tokens]
    empty = sum(1 for t in content_tokens if not t)
    log(f"[data] Content_clean xong, {empty} khach san rong sau lam sach")

    # cot phuc vu danh gia (khop notebook Request1) va insight (khop notebook Request3)
    df1["Star_Rank"] = df1["Hotel_Rank"].apply(extract_star_rank)
    df1["City"] = df1["Hotel_Address"].apply(
        lambda a: ([x.strip() for x in str(a).split(",")][-2]
                   if len([x.strip() for x in str(a).split(",")]) >= 2
                   else [x.strip() for x in str(a).split(",")][-1]))
    kw = ["Căn hộ", "Chung cư", "Nhà riêng", "Biệt thự", "Nhà mặt đất"]
    df1["Type"] = df1["Hotel_Name"].apply(
        lambda n: "Căn hộ/Nhà" if any(k in str(n) for k in kw) else "Khách sạn")

    # binh luan
    df_cmt = pd.DataFrame()
    if os.path.exists(cmt_path):
        df_cmt = pd.read_csv(cmt_path)
        df_cmt.columns = [str(c).strip().replace(" ", "_") for c in df_cmt.columns]
        log(f"[data] hotel_comments: {df_cmt.shape} | cot: {list(df_cmt.columns)}")
        if "Reviewer_ID" not in df_cmt.columns and "Reviewer_Name" in df_cmt.columns:
            df_cmt["Reviewer_ID"] = df_cmt["Reviewer_Name"].astype(str).str.strip().astype("category").cat.codes
            log(f"[data] sinh Reviewer_ID: {df_cmt['Reviewer_ID'].nunique()} nguoi dung")
        if "Score" in df_cmt.columns:
            df_cmt["Score"] = df_cmt["Score"].apply(convert_comma_number)

        # Parse ngay danh gia (khop notebook Request3): "Đã nhận xét vào 30 tháng 7 2023"
        if "Review_Date" in df_cmt.columns:
            df_cmt["Review_Date_parsed"] = df_cmt["Review_Date"].apply(parse_review_date)
            n_ok = df_cmt["Review_Date_parsed"].notna().sum()
            log(f"[data] parse ngay danh gia: {n_ok}/{len(df_cmt)} thanh cong")
            df_cmt["Review_Year"] = df_cmt["Review_Date_parsed"].dt.year
            df_cmt["Review_YearMonth"] = df_cmt["Review_Date_parsed"].dt.to_period("M")

        # comments_count: dung cot co san trong hotel_info neu co, khong thi tu tinh
        if "comments_count" not in df1.columns:
            cnt = df_cmt.groupby("Hotel_ID").size().rename("comments_count")
            df1 = df1.merge(cnt, on="Hotel_ID", how="left")
            df1["comments_count"] = df1["comments_count"].fillna(0).astype(int)
            log("[data] tu tinh comments_count tu hotel_comments")
    else:
        log(f"[canh bao] khong thay {cmt_path} -> Tab 2 va Tab 3 se thieu du lieu")
        if "comments_count" not in df1.columns:
            df1["comments_count"] = 0

    os.makedirs(DATA_DIR, exist_ok=True)
    df1.to_pickle(os.path.join(DATA_DIR, "hotel_info_clean.pkl"))
    if not df_cmt.empty:
        df_cmt.to_pickle(os.path.join(DATA_DIR, "hotel_comments_clean.pkl"))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"all_stopwords": all_stopwords, "en_vn_dict": en_vn_dict,
         "english_stopwords": ENGLISH_STOPWORDS, "special_chars": SPECIAL_CHARS},
        os.path.join(MODEL_DIR, "text_resources.joblib"),
    )
    log("[data] da luu data/*.pkl va models/text_resources.joblib")
    return df1, df_cmt, content_tokens


# ============================ 2. CONTENT-BASED ============================
def build_content(df1, content_tokens, no_below=5, no_above=0.5):
    from gensim import corpora, models, similarities
    from scipy.sparse import save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Gensim ---
    dictionary = corpora.Dictionary(content_tokens)
    log(f"[gensim] tu vung truoc filter: {len(dictionary.token2id)}")
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    log(f"[gensim] tu vung sau filter : {len(dictionary.token2id)}")

    corpus = [dictionary.doc2bow(t) for t in content_tokens]
    tfidf_gensim = models.TfidfModel(corpus)
    index_gensim = similarities.SparseMatrixSimilarity(
        tfidf_gensim[corpus], num_features=len(dictionary.token2id))
    gensim_sim = np.array([index_gensim[tfidf_gensim[b]] for b in corpus], dtype=np.float32)

    dictionary.save(os.path.join(MODEL_DIR, "dictionary.pkl"))
    tfidf_gensim.save(os.path.join(MODEL_DIR, "tfidf_gensim.pkl"))
    index_gensim.save(os.path.join(MODEL_DIR, "sparse_index.pkl"))
    np.save(os.path.join(MODEL_DIR, "gensim_sim.npy"), gensim_sim)
    log(f"[gensim] ma tran similarity: {gensim_sim.shape}")

    # --- Cosine ---
    vectorizer = TfidfVectorizer(analyzer="word")
    tfidf_matrix = vectorizer.fit_transform(df1["Content_clean"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    save_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"), tfidf_matrix)
    np.save(os.path.join(MODEL_DIR, "cosine_sim.npy"), cosine_sim)
    log(f"[cosine] tu vung {len(vectorizer.vocabulary_)}, ma tran {cosine_sim.shape}")

    return gensim_sim, cosine_sim, dictionary, tfidf_gensim, index_gensim, vectorizer, tfidf_matrix


# ============================ 3. DANH GIA (khop notebook) ============================
def evaluate(df1, content_tokens, gensim_sim, cosine_sim,
             dictionary, tfidf_gensim, index_gensim, vectorizer, tfidf_matrix,
             n_sample=100, nums=5):
    from sklearn.metrics.pairwise import cosine_similarity

    def evaluate_by_id(sim, name):
        np.random.seed(42)
        pool = df1[df1["Star_Rank"].notna()].index.to_numpy()
        idxs = np.random.choice(pool, min(n_sample, len(pool)), replace=False)
        sd, cr, tr, allrec = [], [], [], set()
        for i in idxs:
            base = df1.iloc[i]
            row = sim[i].copy()
            row[i] = -np.inf
            top = np.argsort(-row)[:nums]
            recs = df1.iloc[top]
            sd.append(np.abs(recs["Star_Rank"] - base["Star_Rank"]).mean())
            cr.append((recs["City"] == base["City"]).mean())
            tr.append((recs["Type"] == base["Type"]).mean())
            allrec.update(top.tolist())
        return {"Model": name, "star_diff": round(np.nanmean(sd), 3),
                "same_city": round(np.mean(cr), 3), "same_type": round(np.mean(tr), 3),
                "coverage": round(len(allrec) / len(df1), 3)}

    def self_retrieval(fn, name):
        np.random.seed(42)
        sample = np.random.choice(len(df1), min(n_sample, len(df1)), replace=False)
        t1 = t5 = 0
        for i in sample:
            words = content_tokens[i][:20]
            if not words:
                continue
            top5 = fn(words)
            if len(top5) and i == top5[0]:
                t1 += 1
            if i in top5:
                t5 += 1
        n = len(sample)
        return {"Model": name, "Top1_Accuracy": round(t1 / n, 3), "Top5_HitRate": round(t5 / n, 3)}

    def g_topk(words):
        sim = index_gensim[tfidf_gensim[dictionary.doc2bow(words)]]
        return np.argsort(sim)[::-1][:5]

    def c_topk(words):
        sim = cosine_similarity(vectorizer.transform([" ".join(words)]), tfidf_matrix).flatten()
        return np.argsort(sim)[::-1][:5]

    by_id = pd.DataFrame([evaluate_by_id(gensim_sim, "Gensim"), evaluate_by_id(cosine_sim, "Cosine")])
    by_search = pd.DataFrame([self_retrieval(g_topk, "Gensim"), self_retrieval(c_topk, "Cosine")])

    log("\n--- Danh gia theo Hotel_ID ---")
    log(by_id.to_string(index=False))
    log("\n--- Danh gia theo search noi dung tu do ---")
    log(by_search.to_string(index=False))

    joblib.dump({"by_id": by_id, "by_search": by_search},
                os.path.join(MODEL_DIR, "eval_metrics.joblib"))


# ============================ 4. COLLABORATIVE ============================
def build_collaborative(df_cmt, df_hotel):
    """
    Item-Based KNN — khop Project1_Request2.ipynb (day la mo hinh duoc chon cuoi cung
    trong 3 mo hinh notebook thu: ALS, Item-Based KNN, Item-Based Cosine).

    Buoc (giong het notebook):
      1) Hotel_Index (theo Hotel_ID xuat hien trong hotel_comments) + User_Index (theo Reviewer_ID)
      2) Pivot thanh ma tran Hotel_Index x User_Index, dien 0 cho o thieu
      3) NearestNeighbors(metric="cosine", algorithm="brute") fit tren ma tran (dang thua, luu sparse)

    QUAN TRONG: notebook tu phat hien co "132 khach san chi co trong hotel_comments,
    khong co trong hotel_info" (muc 5.6 Nhan xet). Neu KNN goi y trung 1 trong so do,
    app se khong co ten/dia chi/anh de hien thi. Vi vay CHI xay KNN tren phan giao
    (Hotel_ID vua co danh gia, vua co metadata trong hotel_info) — dung "Intersection:
    341 khach san" ma notebook da tinh — de moi ket qua KNN tra ve deu hien thi duoc.

    Luu y quan trong khac (notebook da tu ghi nhan trong phan Han che):
      Moi Reviewer_ID trong du lieu chi xuat hien DUNG 1 lan (moi nguoi dung chi danh gia
      1 khach san duy nhat). Vi vay khong co User_Index nao co gia tri > 0 o 2 hang
      (2 khach san) cung luc -> cosine similarity giua 2 khach san BAT KY se luon = 0.
      Mo hinh van huan luyen va chay duoc (dung yeu cau ky thuat), nhung ket qua goi y
      thuc chat la ngau nhien giua cac khach san co cung khoang cach = 1 (khong tim thay
      tin hieu tuong dong nao). Day la han che cua CHINH DU LIEU, khong phai loi code.
    """
    if df_cmt.empty or "Reviewer_ID" not in df_cmt.columns or "Score" not in df_cmt.columns:
        log("[cf] thieu du lieu -> bo qua Collaborative filtering")
        return
    from scipy.sparse import csr_matrix, save_npz
    from sklearn.neighbors import NearestNeighbors

    r = df_cmt[["Reviewer_ID", "Hotel_ID", "Score"]].dropna()

    # Chi giu Hotel_ID vua co danh gia, vua co mat trong hotel_info (co the hien thi duoc)
    valid_hotel_ids = set(df_hotel["Hotel_ID"])
    n_before = r["Hotel_ID"].nunique()
    r = r[r["Hotel_ID"].isin(valid_hotel_ids)]
    n_after = r["Hotel_ID"].nunique()
    if n_after < n_before:
        log(f"[cf] loai {n_before - n_after} khach san chi co danh gia nhung khong co trong "
            f"hotel_info (khong the hien thi) -> con {n_after} khach san hop le")

    if r.empty:
        log("[cf] khong co danh gia hop le sau khi loc -> bo qua")
        return

    # Hotel_Index: theo thu tu Hotel_ID xuat hien (giong hotel_mapping trong notebook)
    hotels = np.sort(r["Hotel_ID"].unique())
    users = np.sort(r["Reviewer_ID"].unique())
    hotel_index = {h: i for i, h in enumerate(hotels)}
    user_index = {u: j for j, u in enumerate(users)}

    rows = r["Hotel_ID"].map(hotel_index).to_numpy()
    cols = r["Reviewer_ID"].map(user_index).to_numpy()
    vals = r["Score"].to_numpy(dtype=np.float32)
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(hotels), len(users)))
    log(f"[cf] ma tran Hotel x User: {matrix.shape}, {matrix.nnz} danh gia, "
        f"do thua {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")

    n_neighbors = min(11, len(hotels))
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    knn.fit(matrix)
    log(f"[cf] KNN da huan luyen tren {len(hotels)} khach san co danh gia")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_npz(os.path.join(MODEL_DIR, "cf_matrix.npz"), matrix)
    joblib.dump(knn, os.path.join(MODEL_DIR, "cf_knn_model.joblib"))
    pd.DataFrame({"Hotel_Index": range(len(hotels)), "Hotel_ID": hotels}) \
        .to_pickle(os.path.join(MODEL_DIR, "cf_hotel_index.pkl"))
    log(f"[cf] da luu cf_matrix.npz, cf_knn_model.joblib, cf_hotel_index.pkl "
        f"({len(hotels)} khach san co the goi y)")


def run_all(info_path=None, cmt_path=None, sw_path=None, envn_path=None,
            top_n=20, do_eval=True):
    """Chay toan bo quy trinh. App goi lai ham nay khi bam nut."""
    info_path = info_path or resolve(HOTEL_INFO_CSV, "hotel_info.csv")
    cmt_path = cmt_path or resolve(HOTEL_COMMENTS_CSV, "hotel_comments.csv")
    sw_path = sw_path or resolve(STOPWORDS_TXT, "vietnamese-stopwords.txt")
    envn_path = envn_path or resolve(EN_VN_TXT, "english-vnmese.txt")

    df1, df_cmt, tokens = build_data(info_path, cmt_path, sw_path, envn_path)
    parts = build_content(df1, tokens)
    if do_eval:
        evaluate(df1, tokens, *parts)
    build_collaborative(df_cmt, df1)
    return df1, df_cmt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default=resolve(HOTEL_INFO_CSV, "hotel_info.csv"))
    ap.add_argument("--cmt", default=resolve(HOTEL_COMMENTS_CSV, "hotel_comments.csv"))
    ap.add_argument("--stopwords", default=resolve(STOPWORDS_TXT, "vietnamese-stopwords.txt"))
    ap.add_argument("--envn", default=resolve(EN_VN_TXT, "english-vnmese.txt"))
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    df1, df_cmt, tokens = build_data(a.info, a.cmt, a.stopwords, a.envn)
    g_sim, c_sim, dct, tf_g, idx_g, vec, tf_m = build_content(df1, tokens)
    if not a.skip_eval:
        evaluate(df1, tokens, g_sim, c_sim, dct, tf_g, idx_g, vec, tf_m)
    build_collaborative(df_cmt, df1)
    log("\nHoan tat. Chay: streamlit run app.py")
