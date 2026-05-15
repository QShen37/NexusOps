@echo off
chcp 65001 >nul
title Milvus Server

echo ========================================
echo   Starting Milvus Vector Database
echo ========================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查 Docker
echo [1/3] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found! Please install Docker Desktop.
    pause
    exit /b 1
)
echo [OK] Docker is available

:: 检查 docker-compose 配置文件
echo [2/3] Checking configuration...
if not exist "milvus-standalone-docker-compose.yml" (
    echo [ERROR] milvus-standalone-docker-compose.yml not found!
    echo Please make sure the file exists in current directory.
    pause
    exit /b 1
)

:: 检查并清理冲突的容器
echo [3/3] Starting Milvus...
docker ps -a | findstr "milvus-minio" >nul
if not errorlevel 1 (
    echo [WARN] Found existing Milvus containers, cleaning up...
    docker stop milvus-minio milvus-etcd milvus-standalone 2>nul
    docker rm milvus-minio milvus-etcd milvus-standalone 2>nul
    echo [OK] Cleanup complete
)

:: 启动 Milvus
docker-compose -f milvus-standalone-docker-compose.yml up -d

echo Waiting for Milvus to initialize...
timeout /t 15 /nobreak >nul

:: 验证连接
echo Verifying Milvus connection...
python -c "from pymilvus import connections; connections.connect(host='localhost', port='19530', timeout=5); print('Connected successfully!')" 2>nul
if errorlevel 1 (
    echo [ERROR] Milvus failed to start properly!
    echo.
    echo Checking container logs...
    docker logs milvus-standalone --tail 20
    echo.
    echo [INFO] You can check container status with: docker ps
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Milvus is running!
echo   - Host: localhost
echo   - Port: 19530
echo   - Collection: biz
echo ========================================
echo.
echo To stop Milvus, run: docker-compose -f milvus-standalone-docker-compose.yml down
echo.
pause