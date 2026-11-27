@echo off
title GERDGuard Launcher
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "run-all.ps1"
