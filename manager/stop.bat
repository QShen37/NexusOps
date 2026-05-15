@echo off
chcp 65001 >nul
title Stop All Services

echo ========================================
echo   Stopping All Services
echo ========================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 1. 停止所有 MCP Servers
echo [1/2] Stopping MCP Servers...
echo.

echo Stopping MCP server processes...
taskkill /f /im python.exe /fi "windowtitle eq CLS Server" 2>nul && echo   [OK] CLS Server stopped || echo   [INFO] CLS Server not running
taskkill /f /im python.exe /fi "windowtitle eq Monitor Server" 2>nul && echo   [OK] Monitor Server stopped || echo   [INFO] Monitor Server not running
taskkill /f /im python.exe /fi "windowtitle eq System Server" 2>nul && echo   [OK] System Server stopped || echo   [INFO] System Server not running
taskkill /f /im python.exe /fi "windowtitle eq Docker Server" 2>nul && echo   [OK] Docker Server stopped || echo   [INFO] Docker Server not running
taskkill /f /im python.exe /fi "windowtitle eq Network Server" 2>nul && echo   [OK] Network Server stopped || echo   [INFO] Network Server not running

:: 额外清理：杀掉可能残留的 mcp_server 相关进程（更安全的方式）
echo.
echo Checking for any remaining MCP processes...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv /nh 2^>nul ^| findstr /i "mcp_server"') do (
    set /a pid=%%~i
    taskkill /pid !pid! /f 2>nul && echo   [OK] Killed orphaned MCP process (PID: !pid!)
)

timeout /t 2 /nobreak >nul

:: 2. 停止 Milvus 及相关容器
echo.
echo [2/2] Stopping Milvus and related containers...

:: 检查 docker-compose 文件是否存在
if exist "milvus-standalone-docker-compose.yml" (
    echo Stopping Milvus using docker-compose...
    docker-compose -f milvus-standalone-docker-compose.yml down
    if errorlevel 1 (
        echo [WARN] docker-compose down failed, trying manual cleanup...
    ) else (
        echo [OK] Milvus containers stopped and removed
    )
) else (
    echo [WARN] milvus-standalone-docker-compose.yml not found
)

:: 手动清理可能残留的 Milvus 容器
echo.
echo Checking for any remaining Milvus containers...
docker ps -a | findstr "milvus" >nul
if not errorlevel 1 (
    echo Found remaining Milvus containers, cleaning up...
    docker stop milvus-standalone milvus-minio milvus-etcd 2>nul
    docker rm milvus-standalone milvus-minio milvus-etcd 2>nul
    echo [OK] Remaining containers removed
) else (
    echo [INFO] No Milvus containers found
)

:: 可选：清理虚拟网络（如果有创建）
echo.
echo Checking for Milvus network...
docker network ls | findstr "milvus" >nul
if not errorlevel 1 (
    echo Removing Milvus network...
    docker network rm milvus 2>nul
    echo [OK] Milvus network removed
)

echo.
echo ========================================
echo [SUCCESS] All services have been stopped!
echo ========================================
echo.
echo Summary:
echo   - MCP Servers: Stopped
echo   - Milvus Database: Stopped
echo   - Containers: Removed
echo.
echo To start all services again, run: start.bat
echo To start individually, run:
echo   - start_milvus.bat
echo   - start_mcp_servers.bat
echo.

pause