# ocr_app.py
"""
Streamlit を使った PDF OCR アプリ

Usage:
 1. 仮想環境を有効化
    Windows:  .venv\\Scripts\\activate
    macOS/Linux: source .venv/bin/activate

 2. 必要パッケージをインストール
    pip install streamlit pymupdf opencv-python-headless pillow easyocr

 3. アプリを起動
    streamlit run ocr_app.py

 4. ブラウザで http://localhost:8501 にアクセスし、PDF をアップロードして OCR
"""
import os
import sys
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import streamlit as st

# 画像の前処理関数
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)
    coords = cv2.findNonZero(denoised)
    if coords is not None:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h, w = denoised.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        denoised = cv2.warpAffine(
            denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    return denoised

# Streamlit アプリ設定
st.set_page_config(page_title="PDF OCR", page_icon="📄")
st.title("PDF OCR アプリ 📄")

# ファイルアップロード UI
uploaded_file = st.file_uploader("PDFファイルをアップロード", type=["pdf"])
if not uploaded_file:
    st.info("まずは PDF ファイルをアップロードしてください。")
    st.stop()

# オプション設定
dpi = st.slider("画像変換 DPI", 100, 600, 200, step=50)
langs = st.multiselect("OCR 言語 (EasyOCR) (例: ja, en)", ["ja", "en"], default=["ja", "en"])

# OCR 実行
if st.button("OCR 実行"):
    with st.spinner("処理中... この処理には時間がかかる場合があります"):
        pdf_bytes = uploaded_file.read()
        # PyMuPDF で PDF を画像に変換
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        doc.close()

        # EasyOCR Reader の初期化（GPU=False）
        reader = easyocr.Reader(langs, gpu=False)
        all_text = []
        for idx, pil_img in enumerate(pages, start=1):
            # PIL→OpenCV
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            proc_img = preprocess_image(img_cv)
            # OCR 実行
            results = reader.readtext(proc_img, detail=0)
            text = "\n".join(results)
            all_text.append(f"--- ページ {idx} ---\n" + text)
        full_text = "\n".join(all_text)

    st.success("OCR 完了！")
    st.text_area("OCR 結果", full_text, height=400)
    # ダウンロードボタン
    out_name = os.path.splitext(uploaded_file.name)[0] + '_output.txt'
    st.download_button("テキストをダウンロード", data=full_text, file_name=out_name)


