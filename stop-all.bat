@echo off
title Hentikan GERDGuard
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "stop-all.ps1"
