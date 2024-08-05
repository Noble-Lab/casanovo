@echo off
setlocal

:: Get the directory of the script
set "SCRIPT_DIR=%~dp0"

:: Run Casanovo sequence command
casanovo sequence ^
    sample_data\sample_preprocessed_spectra.mgf ^
    -c "%SCRIPT_DIR%ting_config.yaml" ^
    -m "%SCRIPT_DIR%tiny_model.ckpt" ^
    -o test

:: Check for errors in jmzTabValidator output
java -jar "%SCRIPT_DIR%jmzTabValidator.jar" --check inFile=test.mztab > result.log
findstr /b "[Error-" result.log > nul
if %ERRORLEVEL% EQU 0 (
    echo mzTab validation failed.
    if exist test.log del test.log
    if exist test.mztab del test.mztab
    exit /b 1
)

:: Cleanup
if exist test.log del test.log
if exist test.mztab del test.mztab

endlocal