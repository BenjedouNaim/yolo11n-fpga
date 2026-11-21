@echo off
rem Builds the C++ reference model with MSVC (Visual Studio 2022 Build Tools).
setlocal
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul || exit /b 1
cd /d %~dp0
if not exist build mkdir build
cl /nologo /O2 /fp:precise /std:c++17 /EHsc /W3 /D_CRT_SECURE_NO_WARNINGS /Fobuild\ /Fe:build\yolo_ref.exe main.cpp model.cpp ops.cpp postprocess.cpp || exit /b 1
echo Built build\yolo_ref.exe
