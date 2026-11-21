@echo off
rem C simulation without Vitis: builds the testbench + IP as plain C++ with MSVC.
rem Usage: build_csim.bat [v0|v1]   (default v1)  ->  build\tb_<ver>.exe
rem (Vitis HLS csim_design runs the same sources; see hls\script.tcl.)
setlocal
set VER=%1
if "%VER%"=="" set VER=v1
set DEFS=
if "%VER%"=="v1" set DEFS=/DCONV_IP_V1
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul || exit /b 1
cd /d %~dp0
if not exist build\%VER% mkdir build\%VER%
cl /nologo /O2 /fp:precise /std:c++17 /EHsc /W3 /wd4068 /wd4102 /D_CRT_SECURE_NO_WARNINGS %DEFS% /Fobuild\%VER%\ /Fe:build\tb_%VER%.exe ^
   tb.cpp ..\src\conv_ip_%VER%.cpp ..\..\cpp_ref\model.cpp ..\..\cpp_ref\ops.cpp ..\..\cpp_ref\postprocess.cpp || exit /b 1
echo Built build\tb_%VER%.exe
