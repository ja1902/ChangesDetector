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
:: 5. Model weights: Meta's DINOv2 backbone + the trained parts in weights\
:: -----------------------------------------------
echo.
echo Setting up model weights...

:: The DINOv2 change models are a trained decoder on Meta's frozen DINOv2
:: ViT-B/14. The decoders and the land-cover head ship in weights\ (research /
:: non-commercial use); the backbone comes from Meta's official release.
set "BACKBONE_URL=https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
set "BACKBONE=%SCRIPT_DIR%\dinov2_vitb14_pretrain.pth"
set "ASSEMBLE=%SCRIPT_DIR%\weights\assemble.py"

:: assemble.py exits 3 when the backbone is needed (a model file is missing).
python "%ASSEMBLE%" "%SCRIPT_DIR%"
set "ASSEMBLE_STATUS=!errorlevel!"
if "!ASSEMBLE_STATUS!"=="3" (
    echo Downloading the DINOv2 ViT-B/14 backbone [346 MB] from Meta [dl.fbaipublicfiles.com]...
    curl -L --fail --progress-bar -o "%BACKBONE%" "%BACKBONE_URL%"
    set "BUILD_OK=0"
    if !errorlevel! equ 0 (
        python "%ASSEMBLE%" "%SCRIPT_DIR%" "%BACKBONE%"
        if !errorlevel! equ 0 set "BUILD_OK=1"
    )
    if "!BUILD_OK!"=="1" (
        echo Model weights: OK
    ) else (
        echo WARNING: The model weights could not be built. Download
        echo          %BACKBONE_URL%
        echo          into %SCRIPT_DIR%, then run:
        echo          python weights\assemble.py . dinov2_vitb14_pretrain.pth
    )
    del "%BACKBONE%" 2>nul
) else if not "!ASSEMBLE_STATUS!"=="0" (
    echo WARNING: weights\assemble.py failed [see above].
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
