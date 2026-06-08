@echo off
chcp 65001 >nul
echo ========================================
echo   语音情感识别系统 - 快速启动脚本
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [错误] 未找到requirements.txt文件
    pause
    exit /b 1
)

echo [1/3] 检查依赖包...
pip show torch >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
) else (
    echo ✓ 依赖已就绪
)

echo.
echo [2/3] 检查数据目录...
if not exist "data" (
    echo 数据目录不存在，正在创建...
    python data_utils.py init
    python data_utils.py generate-samples
) else (
    echo ✓ 数据目录已存在
)

echo.
echo ========================================
echo   请选择操作:
echo ========================================
echo   [1] 训练模型
echo   [2] 预测音频情感 (需要提供音频文件路径)
echo   [3] 可视化音频特征 (需要提供音频文件路径)
echo   [4] 评估模型性能
echo   [5] 查看系统信息
echo   [6] 退出
echo ========================================
set /p choice=请输入选项 (1-6):

if "%choice%"=="1" goto train
if "%choice%"=="2" goto predict
if "%choice%"=="3" goto visualize
if "%choice%"=="4" goto evaluate
if "%choice%"=="5" goto info
if "%choice%"=="6" goto end

echo 无效选项
pause
exit /b 1

:train
echo.
echo 开始训练模型...
python main.py train
goto end

:predict
echo.
set /p audio_path=请输入音频文件路径:
if not exist "%audio_path%" (
    echo [错误] 文件不存在: %audio_path%
    pause
    exit /b 1
)
python main.py predict "%audio_path%"
goto end

:visualize
echo.
set /p audio_path=请输入音频文件路径:
if not exist "%audio_path%" (
    echo [错误] 文件不存在: %audio_path%
    pause
    exit /b 1
)
python main.py visualize "%audio_path%"
goto end

:evaluate
echo.
python main.py evaluate
goto end

:info
echo.
echo ========================================
echo   系统信息
echo ========================================
echo   Python版本:
python --version
echo.
echo   PyTorch版本:
python -c "import torch; print(torch.__version__)"
echo.
echo   CUDA可用:
python -c "import torch; print('是' if torch.cuda.is_available() else '否')"
if "%CUDA_VISIBLE_DEVICES%"=="" (
    echo   GPU设备: CPU
) else (
    echo   GPU设备: %CUDA_VISIBLE_DEVICES%
)
echo.
echo   项目目录:
cd
echo.
echo   支持的情感类别:
echo   - angry (愤怒)
echo   - happy (快乐)
echo   - sad (悲伤)
echo   - fearful (恐惧)
echo   - surprise (惊讶)
echo   - neutral (中性)
echo ========================================
pause
goto end

:end
echo.
echo 操作完成！
pause