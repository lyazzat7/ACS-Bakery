@echo off
pyinstaller --noconfirm ^
            --onedir ^
            --console ^
            --clean ^
            --log-level "INFO" ^
            --name "aisu" ^
            --paths "./src" ^
            --paths "./ui" ^
            --paths "./util" ^
            --add-data "./icons;icons" ^
            main.py
pause
