@echo off
setlocal enabledelayedexpansion

:: 脚本信息
echo.
echo ============ 批量数据转换工具 v3.0 ============
echo.

:: 检查参数
if "%~1"=="" (
    echo 使用方法：拖放一个或多个CSV文件到本程序图标
    echo.
    pause
    exit /b 1
)

:: 检查Python环境
where python >nul 2>&1 || (
    echo 错误：未找到Python解释器
    echo 请确认已安装Python并添加至系统PATH环境变量
    echo.
    pause
    exit /b 1
)

:: 创建总输出目录
set "OUT_ROOT=%~dp0table"
if not exist "%OUT_ROOT%" (
    md "%OUT_ROOT%" || (
        echo 无法创建输出目录：%OUT_ROOT%
        pause
        exit /b 1
    )
)

:: 遍历所有拖放的文件
for %%F in (%*) do (
    set "input_file=%%~fF"
    echo.
    echo 正在处理文件：%%~nxF

    :: 验证文件
    if not exist "!input_file!" (
        echo 文件不存在：%%~nxF
        continue
    )
    if /i "%%~xF" neq ".csv" (
        echo 忽略非CSV文件：%%~nxF
        continue
    )

    :: 创建专属输出目录
    set "out_dir=%OUT_ROOT%\%%~nF"
    if not exist "!out_dir!" (
        md "!out_dir!" || (
            echo 无法创建输出目录：%%~nF
            continue
        )
    )

    :: 执行转换
    python "%~dp0data_define.py" "!input_file!" "!out_dir!"
    if !errorlevel! equ 0 (
        echo 转换成功 ^(代码!errorlevel!^)
    ) else (
        echo 转换失败 ^(代码!errorlevel!^)
    )
)

:: 完成提示
echo.
echo 所有文件处理完成
echo 输出目录：%OUT_ROOT%
echo.
pause
