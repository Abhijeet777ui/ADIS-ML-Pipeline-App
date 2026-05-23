@echo off
echo ===================================================
echo Starting ADIS AutoML Dashboard...
echo ===================================================
call .venv\Scripts\activate.bat
streamlit run app.py
pause
