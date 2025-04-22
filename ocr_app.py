# ocr_app.py

"""
Streamlit を使った PDF OCR アプリ

Usage:
 1. 仮想環境を有効化
    Windows:  .venv\\Scripts\\activate
    macOS/Linux: source .venv/bin/activate

 2. 必要パッケージをインストール
    pip install streamlit pymupdf opencv-python-headless pillow pytesseract

 3. アプリを起動
    * 通常 Python版: streamlit run ocr_app.py
    * EXE版: dist\ocr_app.exe を実行するだけ

 4. ブラウザで http://localhost:8501 にアクセスし、PDF をアップロードして OCR
"""
import os
import sys
import cv2
import numpy as np
import fitz  # PyMuPDF を使う
from PIL import Image
import pytesseract
import streamlit as st

# 必要に応じて tesseract_cmd を直接指定 (PATH 未設定時)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
        return cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return denoised

# --- Streamlit アプリ ---
st.set_page_config(page_title="PDF OCR", page_icon="📄")
st.title("PDF OCR アプリ 📄")

uploaded_file = st.file_uploader("PDFファイルをアップロード", type=["pdf"])
if not uploaded_file:
    st.info("まずは PDF ファイルをアップロードしてください。")
    st.stop()

dpi = st.slider("画像変換 DPI", 100, 600, 200, step=50)
langs = st.multiselect("OCR 言語", ["jpn", "eng"], default=["jpn", "eng"])

if st.button("OCR 実行"):
    with st.spinner("処理中..."):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for i in range(doc.page_count):
          page = doc.load_page(i)
          pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
          img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
          pages.append(img)
        doc.close()
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

# --- EXE 版で実行されたときのスペシャル起動 ---
if getattr(sys, 'frozen', False):
    # PyInstaller で生成された EXE として起動された場合
    # 開発モード設定をオフにしてポート指定を有効化
    import os
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    import streamlit.web.cli as stcli
    # Streamlit CLI 引数を組み替え
    sys.argv = [
        'streamlit',
        'run',
        os.path.abspath(sys.executable),
        '--server.port=8501',
        '--server.headless=true'
    ]
    sys.exit(stcli.main())
# ------------------------------------------
