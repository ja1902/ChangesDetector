@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  ChangeDetection QGIS Plugin Installer (Windows)
echo ============================================
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: -----------------------------------------------
:: 1. Check prerequisites
:: -----------------------------------------------
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH. Please install Python 3.10+.
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%i
echo Found Python %PY_VER%

:: -----------------------------------------------
:: 2. Create virtual environment
:: -----------------------------------------------
set "VENV_DIR=%SCRIPT_DIR%\venv"
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists at %VENV_DIR%
) else (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip

:: -----------------------------------------------
:: 3. Install PyTorch (GPU or CPU)
:: -----------------------------------------------
echo.
echo Detecting GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected. Installing PyTorch CPU-only...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: -----------------------------------------------
:: 4. Install Python dependencies
:: -----------------------------------------------
echo.
echo Installing dependencies...
pip install -r "%SCRIPT_DIR%\requirements.txt"

echo Installing mmcv...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
) else (
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.4/index.html
)

:: -----------------------------------------------
:: 5. Download model weights from GitHub Releases
:: -----------------------------------------------
echo.
echo Downloading model weights...

set "GITHUB_RELEASE=https://github.com/ja1902/ChangeDetection/releases/download/v0.1.0"

set "LEVIR_WEIGHTS=%SCRIPT_DIR%\MambaBCD_Small_LEVIRCD+.pth"
if exist "%LEVIR_WEIGHTS%" (
    echo LEVIR-CD+ weights already exist, skipping download.
) else (
    echo Downloading MambaBCD-Small [LEVIR-CD+, ~207 MB]...
    curl -L --fail --progress-bar -o "%LEVIR_WEIGHTS%" "%GITHUB_RELEASE%/MambaBCD_Small_LEVIRCD+.pth"
    if !errorlevel! neq 0 (
        del "%LEVIR_WEIGHTS%" 2>nul
        echo WARNING: Download failed. Please download manually.
        echo          Place the file at: %LEVIR_WEIGHTS%
    )
)

set "SYSU_WEIGHTS=%SCRIPT_DIR%\MambaBCD_Small_SYSU.pth"
if exist "%SYSU_WEIGHTS%" (
    echo SYSU weights already exist, skipping download.
) else (
    echo Downloading MambaBCD-Small [SYSU, ~207 MB]...
    curl -L --fail --progress-bar -o "%SYSU_WEIGHTS%" "%GITHUB_RELEASE%/MambaBCD_Small_SYSU.pth"
    if !errorlevel! neq 0 (
        del "%SYSU_WEIGHTS%" 2>nul
        echo WARNING: Download failed. Please download manually.
        echo          Place the file at: %SYSU_WEIGHTS%
    )
)

set "PEFTCD_LEVIR=%SCRIPT_DIR%\PeftCD_LEVIRCD.ckpt"
if exist "%PEFTCD_LEVIR%" (
    echo PeftCD LEVIR-CD+ weights already exist, skipping download.
) else (
    echo Downloading PeftCD [LEVIR-CD+]...
    curl -L --fail --progress-bar -o "%PEFTCD_LEVIR%" "%GITHUB_RELEASE%/PeftCD_LEVIRCD.ckpt"
    if !errorlevel! neq 0 (
        del "%PEFTCD_LEVIR%" 2>nul
        echo WARNING: Download failed. Please download manually.
        echo          Place the file at: %PEFTCD_LEVIR%
    )
)

set "PEFTCD_SYSU=%SCRIPT_DIR%\PeftCD_SYSU.ckpt"
if exist "%PEFTCD_SYSU%" (
    echo PeftCD SYSU weights already exist, skipping download.
) else (
    echo Downloading PeftCD [SYSU]...
    curl -L --fail --progress-bar -o "%PEFTCD_SYSU%" "%GITHUB_RELEASE%/PeftCD_SYSU.ckpt"
    if !errorlevel! neq 0 (
        del "%PEFTCD_SYSU%" 2>nul
        echo WARNING: Download failed. Please download manually.
        echo          Place the file at: %PEFTCD_SYSU%
    )
)

:: -----------------------------------------------
:: 6. Write environment config for plugin
:: -----------------------------------------------
echo.
echo Writing environment config...
set "ENV_CONFIG=%SCRIPT_DIR%\uchange_qgis_plugin\_env_config.py"
python -c "import pathlib; pathlib.Path(r'%ENV_CONFIG%').write_text('VENV_SITE_PACKAGES = r\"%VENV_DIR%\\Lib\\site-packages\"\n')"
echo   Written: _env_config.py

:: -----------------------------------------------
:: 7. Install plugin to QGIS
:: -----------------------------------------------
echo.
set "QGIS_PLUGINS=%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins"
if not exist "%QGIS_PLUGINS%" mkdir "%QGIS_PLUGINS%"

set "PLUGIN_LINK=%QGIS_PLUGINS%\uchange_qgis_plugin"
if exist "%PLUGIN_LINK%" (
    rmdir "%PLUGIN_LINK%" 2>nul
    del "%PLUGIN_LINK%" 2>nul
)
mklink /J "%PLUGIN_LINK%" "%SCRIPT_DIR%\uchange_qgis_plugin"
if %errorlevel% equ 0 (
    echo Plugin linked to: %PLUGIN_LINK%
) else (
    echo WARNING: Could not create junction. Copying plugin instead...
    xcopy "%SCRIPT_DIR%\uchange_qgis_plugin" "%PLUGIN_LINK%\" /E /I /Y >nul
    echo Plugin copied to: %PLUGIN_LINK%
)

:: -----------------------------------------------
:: Done
:: -----------------------------------------------
echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo Next steps:
echo   1. Open QGIS
echo   2. Go to Plugins ^> Manage and Install Plugins
echo   3. Enable 'ChangeDetection'
echo   4. Find it under Plugins ^> ChangeDetection menu
echo.
pause
