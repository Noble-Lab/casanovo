@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

casanovo sequence sample_data\sample_preprocessed_spectra.mgf -c "%SCRIPT_DIR%ting_config.yaml" -m "%SCRIPT_DIR%tiny_model.ckpt" -o test

java -jar "%SCRIPT_DIR%jmzTabValidator.jar" --check inFile=test.mztab > val.txt
type val.txt
findstr /R "^\[Error-" val.txt >nul
if %errorlevel% equ 0 (
    echo mzTab validation failed.
    if exist test.log del test.log
    if exist test.mztab del test.mztab
    if exist val.txt del val.txt
    exit /b 1
)

echo mzTab validation succedded.
if exist test.log del test.log
if exist test.mztab del test.mztab
if exist val.txt del val.txt

endlocal