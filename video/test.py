import os
import io
import subprocess
import time
source = '../demos/demoimages/IMG_2567.MP4images/'
for root, dirs, filenames in os.walk(source):
    for f in filenames:
        prep=["python3",
                "detect.py",
                "--source", source+f]
                #ha nincs futó process akkor elindul
        subprocess.Popen(prep)
        print(f)
        time.sleep(10)