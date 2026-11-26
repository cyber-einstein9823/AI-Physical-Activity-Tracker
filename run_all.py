"""Run Task 1 preparation and signal plots. Train/test are separate explicit commands."""
import subprocess,sys
cmds=[[sys.executable,'src/load_data.py'],[sys.executable,'src/prepare.py'],[sys.executable,'src/plot_windows.py'],[sys.executable,'src/model.py']]
for c in cmds: subprocess.run(c,check=True)
print('\nTask 1 checks completed. Next: python src/train.py; then python src/test_model.py')
