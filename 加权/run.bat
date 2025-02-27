@echo off
setlocal enabledelayedexpansion

:: 脚本头信息
echo.
echo ============ 批量加权脚本v1 ============
echo.

:: 无参数处理
if "%~1"=="" (
    echo 请将Excel文件拖放到此bat文件上运行
    echo.
    pause
    exit /b 1
)

:: 验证Python环境
where python >nul 2>&1 || (
    echo 错误：未检测到Python环境！
    echo 请先安装Python并添加到系统PATH变量
    echo.
    pause
    exit /b 1
)

:: 自动安装依赖
echo 正在检查Python依赖...
python -m pip install pandas numpy tqdm openpyxl xlrd --user >nul 2>&1
if %errorlevel% neq 0 (
    echo 依赖安装失败，请手动执行：
    echo python -m pip install pandas numpy tqdm openpyxl xlrd
    echo.
    pause
    exit /b 1
)

:: 处理所有拖放文件
for %%F in (%*) do (
    set "input_path=%%~fF"
    set "output_path=%%~dpnF_output.xlsx"
    
    echo.
    echo 正在处理文件：%%~nxF
    
    :: 文件验证
    if not exist "!input_path!" (
        echo 文件不存在：%%~nxF
        continue
    )
    if /i not "%%~xF" == ".xlsx" (
        echo 仅支持.xlsx文件：%%~nxF
        continue
    )
    
    :: 执行Python脚本
    python "%~dp0weight_auto.py" "!input_path!" "!output_path!"
    
    :: 结果处理
    if !errorlevel! equ 0 (
        echo 生成结果文件：%%~nF_output.xlsx
    ) else (
        echo 处理失败（错误码!errorlevel!）
    )
)

:: 完成提示
echo.
echo 所有文件处理完成
echo 输出文件与原始文件同目录
echo.
pause
