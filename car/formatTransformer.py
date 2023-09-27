from PIL import Image
import glob
import os

curPath = os.path.abspath('.')
for gif in glob.glob('*.gif'):
    f, e = os.path.splitext(gif)
    outFile = f + '.png'
    gifPath = os.path.join(curPath, gif)
    Image.open(gifPath).save(outFile)
