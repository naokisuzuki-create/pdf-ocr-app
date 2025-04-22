# ocr_app.py
"""
Streamlit を使った PDF OCR アプリ

Usage:
 1. 仮想環境を有効化
    Windows:  .venv\\Scripts\\activate
    macOS/Linux: source .venv/bin/activate

 2. 必要パッケージをインストール
    pip install streamlit pdf2image opencv-python pillow pytesseract

 3. アプリを起動
    streamlit run ocr_app.py

 4. ブラウザで http://localhost:8501 にアクセスし、PDF をアップロードして OCR
"""
import os
import sys
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
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
        denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
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
langs = st.multiselect("OCR 言語", ["jpn", "eng"], default=["jpn", "eng"])

# OCR 実行
if st.button("OCR 実行"):
    with st.spinner("処理中..."):
        pdf_bytes = uploaded_file.read()
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        texts = []
        for i, pil_img in enumerate(pages, start=1):
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            proc = preprocess_image(img_cv)
            texts.append(f"--- ページ {i} ---\n" + pytesseract.image_to_string(proc, lang='+'.join(langs)))
        result_text = "\n".join(texts)
    st.success("OCR 完了！")
    st.text_area("OCR結果", result_text, height=400)
    out_file = os.path.splitext(uploaded_file.name)[0] + '_output.txt'
    st.download_button("テキストをダウンロード", data=result_text, file_name=out_file)

# EXE 版で実行されたときの処理
if getattr(sys, 'frozen', False):
    import os
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    import streamlit.web.cli as stcli
    sys.argv = [
        'streamlit',
        'run',
        os.path.abspath(sys.executable),
        '--server.port=8501',
        '--server.headless=true'
    ]
    sys.exit(stcli.main())
