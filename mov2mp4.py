import subprocess
import os


if __name__ == '__main__':

    src = r'/home/adlane/projets/twinit-dataset/toconvert'
    dst = r'/home/adlane/projets/twinit-dataset/toconvert/'

    for root, dirs, filenames in os.walk(src, topdown=False):
        #print(filenames)
        for filename in filenames:
            print('[INFO] 1',filename)
            try:
                _format = ''
                if ".flv" in filename.lower():
                    _format=".flv"
                if ".mp4" in filename.lower():
                    _format=".mp4"
                if ".avi" in filename.lower():
                    _format=".avi"
                if ".mov" in filename.lower():
                    _format=".mov"

                inputfile = os.path.join(root, filename)
                print('[INFO] 1',inputfile)
                outputfile = os.path.join(dst, filename.lower().replace(_format, ".mp4"))
                subprocess.call(['ffmpeg', '-i', inputfile, outputfile])
            except:
                print("An exception occurred")