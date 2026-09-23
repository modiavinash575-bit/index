import io, re, math
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

st.set_page_config(page_title="AnswerLens", page_icon="✍️", layout="wide")

RUBRIC = [
    {"id":"concept","name":"Core concept","points":4,"keywords":["definition","principle","concept","because","therefore"]},
    {"id":"reasoning","name":"Reasoning","points":3,"keywords":["explain","reason","derive","step","thus","hence"]},
    {"id":"evidence","name":"Evidence / working","points":2,"keywords":["example","calculation","equation","evidence","data"]},
    {"id":"clarity","name":"Clarity","points":1,"keywords":["conclusion","answer","result","final"]},
]

@st.cache_resource(show_spinner=False)
def load_ocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    return processor, model

def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.35)
    gray = gray.filter(ImageFilter.MedianFilter(3))
    return gray.convert("RGB")

def line_crops(img):
    # Lightweight layout fallback: detect horizontal writing bands.
    a=np.array(img.convert("L"))
    ink=255-a
    row=(ink.mean(axis=1)>10).astype(np.uint8)
    bands=[]; start=None; gap=0
    for i,v in enumerate(row):
        if v:
            if start is None: start=i
            gap=0
        elif start is not None:
            gap+=1
            if gap>12:
                if i-start>8: bands.append((max(0,start-5),min(a.shape[0],i+5)))
                start=None; gap=0
    if start is not None and len(a)-start>8: bands.append((max(0,start-5),len(a)))
    if not bands: return [img]
    return [img.crop((0,y1,img.width,y2)) for y1,y2 in bands]

def transcribe(img):
    processor, model = load_ocr()
    import torch
    crops=line_crops(img)
    texts=[]
    with torch.no_grad():
        for crop in crops:
            px=processor(images=crop, return_tensors="pt").pixel_values
            ids=model.generate(px, max_new_tokens=96)
            texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return "\n".join(t for t in texts if t)

def score_answer(text):
    low=text.lower()
    results=[]; total=0
    for r in RUBRIC:
        hits=[k for k in r["keywords"] if k in low]
        coverage=min(1.0,len(hits)/max(1,min(2,len(r["keywords"]))))
        pts=round(r["points"]*coverage,1)
        total+=pts
        results.append((r["name"],pts,r["points"],hits))
    return total,results

def feedback(results):
    out=[]
    for name,pts,maxpts,hits in results:
        if pts < maxpts:
            out.append(f"**{name}:** add clearer evidence for this criterion.")
        else:
            out.append(f"**{name}:** criterion is represented in the extracted answer.")
    return "\n\n".join(out)

st.title("✍️ AnswerLens")
st.caption("Layout-aware handwritten answer understanding • rubric-aware scoring • grounded feedback")

with st.sidebar:
    st.header("Assessment")
    st.write("Upload a handwritten answer. The system segments writing lines, runs handwritten OCR, maps evidence to a rubric, and flags uncertain cases for review.")
    st.divider()
    st.metric("Rubric points", sum(x["points"] for x in RUBRIC))
    st.caption("This prototype does not claim to replace examiner judgment.")

tab1,tab2,tab3=st.tabs(["Assess answer","Rubric","How it works"])

with tab1:
    up=st.file_uploader("Upload a handwritten answer",type=["png","jpg","jpeg","webp"])
    if up:
        img=Image.open(io.BytesIO(up.read()))
        col1,col2=st.columns(2)
        with col1:
            st.image(img,caption="Original",use_container_width=True)
        clean=preprocess(img)
        with col2:
            st.image(clean,caption="Preprocessed",use_container_width=True)
        if st.button("Analyze answer",type="primary",use_container_width=True):
            with st.spinner("Running layout analysis and handwriting recognition…"):
                try:
                    text=transcribe(clean)
                    total,results=score_answer(text)
                    confidence=max(0.0,min(1.0,0.45+0.08*len(text.split())))
                except Exception as e:
                    st.error("The OCR model could not be loaded or run in this environment.")
                    st.exception(e)
                    st.stop()
            st.subheader("Extracted answer")
            st.text_area("Recognized handwriting",text,height=180)
            a,b,c=st.columns(3)
            a.metric("Rubric score",f"{total:g}/10")
            b.metric("Criteria covered",f"{sum(pts==mx for _,pts,mx,_ in results)}/{len(results)}")
            c.metric("Review status","Human review" if confidence<0.7 else "Ready for review")
            st.subheader("Rubric evidence")
            for name,pts,mx,hits in results:
                st.write(f"**{name} — {pts:g}/{mx}**")
                st.progress(min(1.0,pts/mx if mx else 0))
                st.caption("Evidence terms: "+(", ".join(hits) if hits else "not detected"))
            st.subheader("Feedback")
            st.markdown(feedback(results))
            if confidence<0.7:
                st.warning("Low-confidence extraction: verify the transcription and mark before using the result.")
    else:
        st.info("Upload a page image to begin.")

with tab2:
    st.subheader("Example rubric")
    for r in RUBRIC:
        st.write(f"**{r['name']} — {r['points']} points**")
        st.caption("Concept signals: "+", ".join(r["keywords"]))

with tab3:
    st.markdown("""
### Pipeline
1. Image normalization
2. Writing-line / layout segmentation
3. Handwritten text recognition (TrOCR)
4. Reading-order reconstruction
5. Rubric-criterion evidence matching
6. Criterion-level score and feedback
7. Confidence gate for human review

### Research evaluation
Use IAM or another handwriting benchmark for CER/WER, then use an annotated answer-marking set for quadratic weighted kappa (QWK), MAE, RMSE and ±1-point accuracy. Keep questions unseen between rubric development and final evaluation.

### Important
The demonstration rubric uses transparent lexical evidence so the site remains inspectable. A research deployment should replace that component with calibrated semantic retrieval plus an independently validated rubric scorer.
""")
