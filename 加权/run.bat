@echo off
setlocal enabledelayedexpansion

:: �ű�ͷ��Ϣ
echo.
echo ============ ������Ȩ�ű�v1 ============
echo.

:: �޲�������
if "%~1"=="" (
    echo �뽫Excel�ļ��Ϸŵ���bat�ļ�������
    echo.
    pause
    exit /b 1
)

:: ��֤Python����
where python >nul 2>&1 || (
    echo ����δ��⵽Python������
    echo ���Ȱ�װPython����ӵ�ϵͳPATH����
    echo.
    pause
    exit /b 1
)

:: �Զ���װ����
echo ���ڼ��Python����...
python -m pip install pandas numpy tqdm openpyxl xlrd --user >nul 2>&1
if %errorlevel% neq 0 (
    echo ������װʧ�ܣ����ֶ�ִ�У�
    echo python -m pip install pandas numpy tqdm openpyxl xlrd
    echo.
    pause
    exit /b 1
)

:: ���������Ϸ��ļ�
for %%F in (%*) do (
    set "input_path=%%~fF"
    set "output_path=%%~dpnF_output.xlsx"
    
    echo.
    echo ���ڴ����ļ���%%~nxF
    
    :: �ļ���֤
    if not exist "!input_path!" (
        echo �ļ������ڣ�%%~nxF
        continue
    )
    if /i not "%%~xF" == ".xlsx" (
        echo ��֧��.xlsx�ļ���%%~nxF
        continue
    )
    
    :: ִ��Python�ű�
    python "%~dp0weight_auto.py" "!input_path!" "!output_path!"
    
    :: �������
    if !errorlevel! equ 0 (
        echo ���ɽ���ļ���%%~nF_output.xlsx
    ) else (
        echo ����ʧ�ܣ�������!errorlevel!��
    )
)

:: �����ʾ
echo.
echo �����ļ��������
echo ����ļ���ԭʼ�ļ�ͬĿ¼
echo.
pause
