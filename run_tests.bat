@echo off
echo ===================================================
echo Running ADIS Automated Test Suite...
echo ===================================================
call .venv\Scripts\activate.bat
pytest tests/ -v
pause
