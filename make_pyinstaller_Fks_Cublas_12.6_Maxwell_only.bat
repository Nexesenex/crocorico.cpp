cd /d "%~dp0"
call create_ver_file.bat
copy "P:\NVIDIAGPUCT\CUDA\v12.6\bin\cudart64_12.dll" .\ /Y
copy "P:\NVIDIAGPUCT\CUDA\v12.6\bin\cublasLt64_12.dll" .\ /Y
copy "P:\NVIDIAGPUCT\CUDA\v12.6\bin\cublas64_12.dll" .\ /Y
PyInstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all psutil --icon "./nikogreen.ico" --add-data "./winclinfo.exe;." --add-data "./OpenCL.dll;." --add-data "./kcpp_adapters;./kcpp_adapters" --add-data "./gguf-py;./gguf-py" --add-data "./koboldcpp.py;." --add-data "./klite.embd;." --add-data "./kcpp_docs.embd;." --add-data "./kcpp_sdui.embd;." --add-data "./taesd.embd;." --add-data "./taesd_xl.embd;." --add-data "./taesd_f.embd;." --add-data "./taesd_3.embd;." --add-data "./koboldcpp_cublas.dll;." --add-data "./cublasLt64_12.dll;." --add-data "./cublas64_12.dll;." --add-data "./cudart64_12.dll;." --add-data "./msvcp140.dll;." --add-data "./msvcp140_codecvt_ids.dll;." --add-data "./vcruntime140.dll;." --add-data "./vcruntime140_1.dll;." --add-data "./rwkv_vocab.embd;." --add-data "./rwkv_world_vocab.embd;." --version-file "./version.txt" "./koboldcpp.py"  -n "croco.cpp_fks_cuda_12.6_Maxwell_only.exe"
pause