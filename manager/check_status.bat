@echo off
chcp 65001 >nul
title MCP Environment Status

echo ========================================
echo   MCP Environment Status
echo ========================================
echo.

:: 检查 Milvus
echo [Milvus Status]
docker ps | findstr "milvus-standalone" >nul
if not errorlevel 1 (
    echo   [RUNNING] Milvus is running
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr "milvus"
) else (
    echo   [STOPPED] Milvus is not running
    echo   To start: run start_milvus.bat
)
echo.

:: 检查 MCP Servers
echo [MCP Servers Status]
set count=0
tasklist /fi "windowtitle eq CLS Server" /fo csv 2>nul | findstr /i "python.exe" >nul
if not errorlevel 1 (echo   [RUNNING] CLS Server) else (echo   [STOPPED] CLS Server & set /a count+=1)

tasklist /fi "windowtitle eq Monitor Server" /fo csv 2>nul | findstr /i "python.exe" >nul
if not errorlevel 1 (echo   [RUNNING] Monitor Server) else (echo   [STOPPED] Monitor Server & set /a count+=1)

tasklist /fi "windowtitle eq System Server" /fo csv 2>nul | findstr /i "python.exe" >nul
if not errorlevel 1 (echo   [RUNNING] System Server) else (echo   [STOPPED] System Server & set /a count+=1)

tasklist /fi "windowtitle eq Docker Server" /fo csv 2>nul | findstr /i "python.exe" >nul
if not errorlevel 1 (echo   [RUNNING] Docker Server) else (echo   [STOPPED] Docker Server & set /a count+=1)

tasklist /fi "windowtitle eq Network Server" /fo csv 2>nul | findstr /i "python.exe" >nul
if not errorlevel 1 (echo   [RUNNING] Network Server) else (echo   [STOPPED] Network Server & set /a count+=1)

if %count%==5 (
    echo.
    echo [INFO] No MCP servers are running
    echo To start: run start_mcp_servers.bat
)
echo.

:: 检查日志文件大小
echo [Log Files]
if exist "logs" (
    dir logs\*.log 2>nul
) else (
    echo   No logs directory found
)
echo.

pause