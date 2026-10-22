@echo off
rem Nesne Silici'yi başlatır. İlk kurulum için: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
cd /d "%~dp0"
".venv\Scripts\python.exe" -m nesne_silici %*
