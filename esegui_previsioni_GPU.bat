@echo off
REM ⚙️ Attiva Conda e l'ambiente tf-gpu
call C:\Users\lucar\miniconda3\Scripts\activate.bat tf-gpu

REM 📍 Spostati nella cartella contenente lo script (modifica se diverso)
cd /d C:\Users\lucar\Desktop\Test_D1

REM 🚀 Esegui gli script con Python (ora l'ambiente ha TF con GPU attiva)
python ml_predictions.py --symbol EURUSD --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 1.12975
python ml_predictions.py --symbol GBPCAD --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 1.83258
python ml_predictions.py --symbol GBPJPY --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 192.346
python ml_predictions.py --symbol GBPUSD --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 1.32693
python ml_predictions.py --symbol NZDCAD --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 0.82030
python ml_predictions.py --symbol NZDUSD --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 0.59388
python ml_predictions.py --symbol USDCAD --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 1.38125
python ml_predictions.py --symbol USDCHF --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 0.82627
python ml_predictions.py --symbol USDJPY --data_folder C:\Users\lucar\Desktop\Test_D1 --next_open 144.957

pause
