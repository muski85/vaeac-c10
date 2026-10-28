@echo off
REM Quick check: imputes one Iris file into a separate test_run folder.
cd /d "%~dp0"
set "ROOT=%USERPROFILE%\Downloads\data_mining_project\Course Project Datasets\Course Project Datasets"
python main.py --data_root "%ROOT%" --datasets Iris --patterns C --rates 1 --out test_run --no_plots
pause
