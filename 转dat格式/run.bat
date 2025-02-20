@echo off
setlocal enabledelayedexpansion

:: �ű���Ϣ
echo.
echo ============ ��������ת������ v3.0 ============
echo.

:: ������
if "%~1"=="" (
    echo ʹ�÷������Ϸ�һ������CSV�ļ���������ͼ��
    echo.
    pause
    exit /b 1
)

:: ���Python����
where python >nul 2>&1 || (
    echo ����δ�ҵ�Python������
    echo ��ȷ���Ѱ�װPython�������ϵͳPATH��������
    echo.
    pause
    exit /b 1
)

:: ���������Ŀ¼
set "OUT_ROOT=%~dp0table"
if not exist "%OUT_ROOT%" (
    md "%OUT_ROOT%" || (
        echo �޷��������Ŀ¼��%OUT_ROOT%
        pause
        exit /b 1
    )
)

:: ���������Ϸŵ��ļ�
for %%F in (%*) do (
    set "input_file=%%~fF"
    echo.
    echo ���ڴ����ļ���%%~nxF

    :: ��֤�ļ�
    if not exist "!input_file!" (
        echo �ļ������ڣ�%%~nxF
        continue
    )
    if /i "%%~xF" neq ".csv" (
        echo ���Է�CSV�ļ���%%~nxF
        continue
    )

    :: ����ר�����Ŀ¼
    set "out_dir=%OUT_ROOT%\%%~nF"
    if not exist "!out_dir!" (
        md "!out_dir!" || (
            echo �޷��������Ŀ¼��%%~nF
            continue
        )
    )

    :: ִ��ת��
    python "%~dp0data_define.py" "!input_file!" "!out_dir!"
    if !errorlevel! equ 0 (
        echo ת���ɹ� ^(����!errorlevel!^)
    ) else (
        echo ת��ʧ�� ^(����!errorlevel!^)
    )
)

:: �����ʾ
echo.
echo �����ļ��������
echo ���Ŀ¼��%OUT_ROOT%
echo.
pause
