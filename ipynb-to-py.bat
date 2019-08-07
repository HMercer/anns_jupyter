@echo off
set /P id=File path: 
@echo %id%.ipynb
jupyter nbconvert --to script %id%
pause