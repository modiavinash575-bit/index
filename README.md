# AnswerLens — Handwritten Answer Understanding

A deployable Streamlit capstone for layout-aware handwritten answer recognition and rubric-aware feedback.

## Deploy

This repository is designed for Streamlit Community Cloud. Connect the repository, choose `streamlit_app.py` as the entrypoint, and deploy.

The app lazily downloads Microsoft's `trocr-base-handwritten` model on the first analysis request.

## Research architecture

Image → preprocessing → layout/line segmentation → TrOCR → structured answer → rubric evidence → criterion score → grounded feedback → confidence/human review.

## Disclaimer

This is an assessment-support prototype, not an autonomous replacement for a qualified examiner.
