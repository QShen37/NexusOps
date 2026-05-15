@echo off
chcp 65001 >nul
title Start All Services

echo ========================================
echo   Starting All Services
echo ========================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 启动 Milvus
echo [1/2] Starting Milvus...
call "%~dp0start_milvus.bat"

:: 等待用户确认 Milvus 启动完成
echo.
echo Press any key to continue starting MCP servers...
pause >nul

:: 启动 MCP Servers
echo [2/2] Starting MCP Servers...
call "%~dp0start_mcp_servers.bat"

echo.
echo ========================================
echo All services are running!
echo ========================================
pause