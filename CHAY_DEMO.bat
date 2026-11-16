@echo off
rem Bam dup file nay de chay demo SPGA goc, GA-TS, GA-SA (Nhom 8)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo.ps1"
if errorlevel 1 pause
