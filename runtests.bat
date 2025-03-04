@echo off
REM ==========================================================
REM ====== Batch Script for leavesFalling Tests ============
REM ==========================================================

REM Define the suffix to append
SET SUFFIX_LEAVES=RainandFog

REM ==========================================================
REM ====== Multi RGB Tests (Input: 40) ======================
REM ==========================================================

echo Running Multi RGB Tests with suffix appended by '%SUFFIX_LEAVES%'

REM Test 1: a2c_naturecnn_multi_rgb
echo 40 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode multi_rgb --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_multi_rgb_obstacle_weather_off\best_model.zip --suffix a2c_naturecnn_multi_rgb_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 1 failed.
    exit /b %ERRORLEVEL%
)
echo Test 1 completed successfully.

REM Test 2: ppo_naturecnn_multi_rgb
echo 40 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode multi_rgb --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_multi_rgb\best_model.zip --suffix ppo_naturecnn_multi_rgb_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 2 failed.
    exit /b %ERRORLEVEL%
)
echo Test 2 completed successfully.

REM ==========================================================
REM ====== Single RGB Tests (Input: 20) =====================
REM ==========================================================

echo Running Single RGB Tests with suffix appended by '%SUFFIX_LEAVES%'

REM Test 3: a2c_naturecnn_single_rgb_obstacle_weather_snowonly
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_single_rgb_obstacle_weather_snowonly\best_model.zip --suffix a2c_naturecnn_single_rgb_obstacle_weather_snowonly_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 3 failed.
    exit /b %ERRORLEVEL%
)
echo Test 3 completed successfully.

REM Test 4: a2c_naturecnn_single_rgb_obstacle_weather_rainandfog
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_single_rgb_obstacle_weather_rainandfog\best_model.zip --suffix a2c_naturecnn_single_rgb_obstacle_weather_rainandfog_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 4 failed.
    exit /b %ERRORLEVEL%
)
echo Test 4 completed successfully.

REM Test 5: a2c_naturecnn_single_rgb_obstacle_weather_leavesfalling
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_single_rgb_obstacle_weather_leavesfalling\best_model.zip --suffix a2c_naturecnn_single_rgb_obstacle_weather_leavesfalling_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 5 failed.
    exit /b %ERRORLEVEL%
)
echo Test 5 completed successfully.

REM Test 6: a2c_naturecnn_single_rgb
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_single_rgb\best_model.zip --suffix a2c_naturecnn_single_rgb_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 6 failed.
    exit /b %ERRORLEVEL%
)
echo Test 6 completed successfully.

REM Test 7: ppo_naturecnn_single_rgb_obstacle_weather_snowonly
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_single_rgb_obstacle_weather_snowonly\best_model.zip --suffix ppo_naturecnn_single_rgb_obstacle_weather_snowonly_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 7 failed.
    exit /b %ERRORLEVEL%
)
echo Test 7 completed successfully.

REM Test 8: ppo_naturecnn_single_rgb_obstacle_weather_rainandfog
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_single_rgb_obstacle_weather_rainandfog\best_model.zip --suffix ppo_naturecnn_single_rgb_obstacle_weather_rainandfog_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 8 failed.
    exit /b %ERRORLEVEL%
)
echo Test 8 completed successfully.

REM Test 9: ppo_naturecnn_single_rgb_obstacle_weather_leavesfalling
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_single_rgb_obstacle_weather_leavesfalling\best_model.zip --suffix ppo_naturecnn_single_rgb_obstacle_weather_leavesfalling_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 9 failed.
    exit /b %ERRORLEVEL%
)
echo Test 9 completed successfully.

REM Test 10: ppo_naturecnn_single_rgb
echo 20 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode single_rgb --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_single_rgb\best_model.zip --suffix ppo_naturecnn_single_rgb_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 10 failed.
    exit /b %ERRORLEVEL%
)
echo Test 10 completed successfully.

REM ==========================================================
REM ====== Depth Tests (Input: 0) ============================
REM ==========================================================

echo Running Depth Tests with suffix appended by '%SUFFIX_LEAVES%'

REM Test 11: a2c_naturecnn_depth_obstacle_weather_snowonly
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_depth_obstacle_weather_snowonly\best_model.zip --suffix a2c_naturecnn_depth_obstacle_weather_snowonly_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 11 failed.
    exit /b %ERRORLEVEL%
)
echo Test 11 completed successfully.

REM Test 12: a2c_naturecnn_depth_obstacle_weather_rainandfog
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_depth_obstacle_weather_rainandfog\best_model.zip --suffix a2c_naturecnn_depth_obstacle_weather_rainandfog_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 12 failed.
    exit /b %ERRORLEVEL%
)
echo Test 12 completed successfully.

REM Test 13: a2c_naturecnn_depth_obstacle_weather_leavesfalling
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\a2c_naturecnn_depth_obstacle_weather_leavesfalling\best_model.zip --suffix a2c_naturecnn_depth_obstacle_weather_leavesfalling_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 13 failed.
    exit /b %ERRORLEVEL%
)
echo Test 13 completed successfully.

REM Test 14: a2c_naturecnn_single_rgb
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_depth\best_model.zip --suffix a2c_naturecnn_depth_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 14 failed.
    exit /b %ERRORLEVEL%
)
echo Test 14 completed successfully.

REM Test 15: ppo_naturecnn_depth_obstacle_weather_snowonly
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_depth_obstacle_weather_snowonly\best_model.zip --suffix ppo_naturecnn_depth_obstacle_weather_snowonly_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 15 failed.
    exit /b %ERRORLEVEL%
)
echo Test 15 completed successfully.

REM Test 16: ppo_naturecnn_depth_obstacle_weather_rainandfog
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_depth_obstacle_weather_rainandfog\best_model.zip --suffix ppo_naturecnn_depth_obstacle_weather_rainandfog_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 16 failed.
    exit /b %ERRORLEVEL%
)
echo Test 16 completed successfully.

REM Test 17: ppo_naturecnn_depth_obstacle_weather_leavesfalling
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_depth_obstacle_weather_leavesfalling\best_model.zip --suffix ppo_naturecnn_depth_obstacle_weather_leavesfalling_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 17 failed.
    exit /b %ERRORLEVEL%
)
echo Test 17 completed successfully.

REM Test 18: ppo_naturecnn_depth
echo 0 | python .\inferenceChecks.py --algorithm PPO --cnn NatureCNN --test_mode depth --test_type sequential --model_path ..\Models\Best\ppo_naturecnn_depth\best_model.zip --suffix ppo_naturecnn_depth_%SUFFIX_LEAVES%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Test 18 failed.
    exit /b %ERRORLEVEL%
)
echo Test 18 completed successfully.

REM ==========================================================
REM ==================== Completion ===========================
REM ==========================================================

echo All leavesFalling tests have been initiated.
pause
