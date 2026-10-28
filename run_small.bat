@echo off
REM Runs our VAEAC on the remaining small datasets. Safe to re-run: finished files are skipped.
cd /d "%~dp0"
set "ROOT=%USERPROFILE%\Downloads\data_mining_project\Course Project Datasets\Course Project Datasets"
python main.py --data_root "%ROOT%" --datasets Ionosphere DERM Difdoug CREDIT BCW PID 4-gauss TTTTEG Yeast Splice CNP Abalone Spam MUSH --out results --no_plots
pause
