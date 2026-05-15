@echo off
:: 设置控制台为 UTF-8 编码
chcp 65001 >nul

title MCP Servers

echo ========================================
echo   Starting MCP Servers
echo ========================================
echo.

:: 切换到 bat 所在目录（manager 文件夹）
cd /d "%~dp0"
echo Bat location: %cd%
echo.

:: mcp_server 在父目录中（和 manager 同级）
set "MCP_PATH=..\mcp_server"
echo Looking for mcp_server at: %MCP_PATH%
echo.

:: 检查 mcp_server 目录
echo [1/5] Checking MCP server files...
if exist "%MCP_PATH%\" (
    echo [OK] mcp_server directory found at %MCP_PATH%
) else (
    echo [ERROR] mcp_server directory not found!
    echo Expected location: %cd%\..\mcp_server\
    echo.
    echo Contents of parent directory:
    dir /b ".."
    echo.
    pause
    exit /b 1
)

:: 检查具体的服务器文件
echo [2/5] Checking server files...
if exist "%MCP_PATH%\cls_server.py" (echo [OK] cls_server.py) else echo [WARN] cls_server.py not found
if exist "%MCP_PATH%\monitor_server.py" (echo [OK] monitor_server.py) else echo [WARN] monitor_server.py not found
if exist "%MCP_PATH%\system_server.py" (echo [OK] system_server.py) else echo [WARN] system_server.py not found
if exist "%MCP_PATH%\docker_server.py" (echo [OK] docker_server.py) else echo [WARN] docker_server.py not found
if exist "%MCP_PATH%\network_server.py" (echo [OK] network_server.py) else echo [WARN] network_server.py not found

:: 检查 Python 环境
echo.
echo [3/5] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)
python --version
echo [OK] Python is available

:: 停止所有 Python 进程（彻底清理）
echo.
echo [4/5] Stopping existing MCP processes...
echo Killing all Python processes...
taskkill /f /im python.exe >nul 2>&1
timeout /t 3 /nobreak >nul
echo [OK] All Python processes stopped

:: 清理日志目录（删除旧的日志文件）
echo.
echo Cleaning log files...
if exist "logs" (
    rmdir /s /q logs 2>nul
    timeout /t 1 /nobreak >nul
)
mkdir logs 2>nul
echo [OK] Log directory cleaned and recreated

:: 启动各个 MCP 服务器
echo.
echo [5/5] Starting MCP Servers...
echo.

set "PROJECT_ROOT=%cd%\.."

:: 使用独立的日志文件，每个服务器一个文件
echo Starting CLS Server...
start "CLS Server" /min cmd /c "cd /d "%PROJECT_ROOT%" && python mcp_server\cls_server.py > "%cd%\logs\cls_server.log" 2>&1"
timeout /t 2 /nobreak >nul

echo Starting Monitor Server...
start "Monitor Server" /min cmd /c "cd /d "%PROJECT_ROOT%" && python mcp_server\monitor_server.py > "%cd%\logs\monitor_server.log" 2>&1"
timeout /t 2 /nobreak >nul

echo Starting System Server...
start "System Server" /min cmd /c "cd /d "%PROJECT_ROOT%" && python mcp_server\system_server.py > "%cd%\logs\system_server.log" 2>&1"
timeout /t 2 /nobreak >nul

echo Starting Docker Server...
start "Docker Server" /min cmd /c "cd /d "%PROJECT_ROOT%" && python mcp_server\docker_server.py > "%cd%\logs\docker_server.log" 2>&1"
timeout /t 2 /nobreak >nul

echo Starting Network Server...
start "Network Server" /min cmd /c "cd /d "%PROJECT_ROOT%" && python mcp_server\network_server.py > "%cd%\logs\network_server.log" 2>&1"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo [SUCCESS] All MCP Servers are starting!
echo ========================================
echo.
echo Project root: %PROJECT_ROOT%
echo Log files: %cd%\logs\
echo.
echo To verify servers are running:
echo   tasklist /fi "imagename eq python.exe"
echo.
echo To view logs:
echo   type logs\cls_server.log
echo.
pause