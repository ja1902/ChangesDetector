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
    for /f "tokens=*" %%m in ('python -c "import sys; print(sys.version_info.minor)"') do set PY_MINOR=%%m
    if !PY_MINOR! GEQ 13 (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    ) else (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    )
    pip install torch torchvision --index-url !TORCH_INDEX!
) else (
    echo No NVIDIA GPU detected. Installing PyTorch CPU-only...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

:: -----------------------------------------------
:: 4. Install Python dependencies
:: -----------------------------------------------
echo.
echo Installing dependencies...
pip install -r "%SCRIPT_DIR%\requirements.txt"

echo Installing AROSICS (co-registration)...
pip install arosics geoarray py_tools_ds shapely scikit-image

:: -----------------------------------------------
:: 5. Download model weights from GitHub Releases
:: -----------------------------------------------
echo.
echo Downloading model weights...

set "GITHUB_RELEASE=https://github.com/ja1902/ChangesDetector/releases/download/v0.8.0"

:: Model weights are for research / non-commercial use (CC BY-NC-SA 4.0 training data).
set "SYNTH_WEIGHTS=%SCRIPT_DIR%\dinov2_vitb14_c2s1_levir.pth"
if exist "%SYNTH_WEIGHTS%" (
    echo DINOv2 synthetic-data weights already exist, skipping download.
) else (
    echo Downloading DINOv2 ViT-B/14 + synthetic data [recommended]...
    curl -L --fail --progress-bar -o "%SYNTH_WEIGHTS%" "%GITHUB_RELEASE%/dinov2_vitb14_c2s1_levir.pth"
    if !errorlevel! neq 0 (
        del "%SYNTH_WEIGHTS%" 2>nul
        echo WARNING: Download failed. Please download dinov2_vitb14_c2s1_levir.pth from
        echo          https://github.com/ja1902/ChangesDetector/releases and place it at: %SYNTH_WEIGHTS%
    )
)

set "LC_HEAD=%SCRIPT_DIR%\landcover_dinov2_vitb14_oem_second.pth"
if exist "%LC_HEAD%" (
    echo Land-cover head weights already exist, skipping download.
) else (
    echo Downloading DINOv2 land-cover head [labelled change]...
    curl -L --fail --progress-bar -o "%LC_HEAD%" "%GITHUB_RELEASE%/landcover_dinov2_vitb14_oem_second.pth"
    if !errorlevel! neq 0 (
        del "%LC_HEAD%" 2>nul
        echo WARNING: Download failed. Please download landcover_dinov2_vitb14_oem_second.pth from
        echo          https://github.com/ja1902/ChangesDetector/releases and place it at: %LC_HEAD%
    )
)

set "LEVIR_WEIGHTS=%SCRIPT_DIR%\dinov2_vitb14_levir.pth"
if exist "%LEVIR_WEIGHTS%" (
    echo DINOv2 generalizable weights already exist, skipping download.
) else (
    echo Downloading DINOv2 ViT-B/14 [generalizable]...
    curl -L --fail --progress-bar -o "%LEVIR_WEIGHTS%" "%GITHUB_RELEASE%/dinov2_vitb14_levir.pth"
    if !errorlevel! neq 0 (
        del "%LEVIR_WEIGHTS%" 2>nul
        echo WARNING: Download failed. Please download dinov2_vitb14_levir.pth from
        echo          https://github.com/ja1902/ChangesDetector/releases and place it at: %LEVIR_WEIGHTS%
    )
)

set "DINOV2_WEIGHTS=%SCRIPT_DIR%\dinov2_vitb14_egybcd.pth"
if exist "%DINOV2_WEIGHTS%" (
    echo DINOv2 fine-tuned weights already exist, skipping download.
) else (
    echo Downloading DINOv2 ViT-B/14 [fine-tuned]...
    curl -L --fail --progress-bar -o "%DINOV2_WEIGHTS%" "%GITHUB_RELEASE%/dinov2_vitb14_egybcd.pth"
    if !errorlevel! neq 0 (
        del "%DINOV2_WEIGHTS%" 2>nul
        echo WARNING: Download failed. Please download dinov2_vitb14_egybcd.pth from
        echo          https://github.com/ja1902/ChangesDetector/releases and place it at: %DINOV2_WEIGHTS%
    )
)

:: -----------------------------------------------
:: 6. Write environment config for plugin
:: -----------------------------------------------
echo.
echo Writing environment config...
set "ENV_CONFIG=%SCRIPT_DIR%\uchange_qgis_plugin\_env_config.py"
python -c "import pathlib; pathlib.Path(r'%ENV_CONFIG%').write_text('VENV_SITE_PACKAGES = r\"%VENV_DIR%\\Lib\\site-packages\"\nVENV_PYTHON = r\"%VENV_DIR%\\Scripts\\python.exe\"\n')"
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
