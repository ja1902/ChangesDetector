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

echo Installing AROSICS (co-registration)...
pip install arosics geoarray py_tools_ds shapely scikit-image

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

set "GITHUB_RELEASE=https://github.com/acoding04/ChangeDetection/releases/download/V0.2.0"

set "CHANGEREX_WEIGHTS=%SCRIPT_DIR%\ChangerEx_r18-512x512_40k_levircd.pth"
if exist "%CHANGEREX_WEIGHTS%" (
    echo ChangerEx weights already exist, skipping download.
) else (
    echo Downloading ChangerEx R18 [LEVIR-CD]...
    curl -L --fail --progress-bar -o "%CHANGEREX_WEIGHTS%" "%GITHUB_RELEASE%/ChangerEx_r18-512x512_40k_levircd.pth"
    if !errorlevel! neq 0 (
        del "%CHANGEREX_WEIGHTS%" 2>nul
        echo WARNING: Download failed. Please download manually.
        echo          Place the file at: %CHANGEREX_WEIGHTS%
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
