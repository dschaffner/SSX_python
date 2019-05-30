#!/usr/local/bin/python
"""Provides useful non-science oriented methods.

Contains several useful methods for navigating directories, etc., specific to the SSX environment."""

# $Id$
# $HeadURL$

# 12/5/08 3:23 PM by Tim Gray
# 297AD2BF-E286-45F5-B2AF-95F12DFF2CF6


__author__ = "Tim Gray"
__version__ = "1.1"

# 1.1 - 2/9/09 - added code so we can tell if we are on ion or elsewhere so we can read data properly

import os
from numpy import array, zeros, arange

import ssxdefaults as ssxdef

# figure sizes - should work in an module that import ssx_py_utils as ssxutil

pubsize=(3.25,2.0)
docsize=(5,2.47)
specsize=(10,4)
displaysize = (8,6)

f = displaysize

def ssxPath(fileName, baseDir = "ssx", dir = None, check = False, mkdir = False):
    """Returns a file path in an intelligent manner.

    If one just fills in the fileName, then returns a full filepath to
    the SSX python directory.  Other default locations can be specified
    by using one word specifiers for other important directories.  If at
    anytime a path is entered into fileName, the baseDir selection is
    over ridden.

    If check is set to True, ssxPath will check for a file's existence as well.

    Possible baseDir choices:
        ssx = /Users/tgray/Documents/SSX/Python
        output = /Users/tgray/Documents/SSX/PythonOutput
        #portal = /Volumes/tgray
        #degas = /Volumes/tgray/degas2/LINUX-cdx/"""
    baseName = os.path.basename(fileName)
    dirName = os.path.dirname(fileName)	
    dirName = os.path.expanduser(dirName)
# 	firstDir = baseDir.split('/')[0]
    if (dirName == '') and baseDir:
        if baseDir == "output":
            if ssxdef.base == '/ssx/':
                dirName = os.path.join(os.path.expanduser('~'), 'PythonOutput')
            else:
                dirName = os.path.join(ssxdef.base, 'PythonOutput')
        elif baseDir == "ssx":
            # dirName = os.path.join(ssxdef.base, "Python")
            dirName = os.path.dirname(ssxdef.__file__)
            if not dirName:
                dirName = os.getcwd()
        elif baseDir == "data":
            dirName = os.path.join(ssxdef.base, "data")
# 		elif firstDir == "ids":
# 			dirName = os.path.join("/Users/tgray/Documents/SSX/PythonOutput/", baseDir)
# 		elif firstDir == "mag":
# 			dirName = os.path.join("/Users/tgray/Documents/SSX/PythonOutput/", baseDir)
        else:
            dirName = ssxdef.base
        if dir:
            dirName = os.path.join(dirName, dir)
    elif (dirName != '') and (baseDir == 'data'):
        dirName = os.path.join(os.path.join(ssxdef.base, 'data'), dirName)
    fileName = os.path.join(dirName, baseName)

    if mkdir:
        dir, trash = os.path.split(fileName)
        os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', dir)
    if check:	
        if os.path.exists(fileName):
            return fileName
        else:
            raise "File '%s' does not exist." % fileName
# 			return ""
    else:
        return fileName

def readFile(fileName, baseDir = "ssx"):
    """Returns lines from specified file."""

# 	fileName = cdxPath(fileName, baseDir)	
# 	if os.path.exists(fileName):
# 		theFile = file(fileName)
# 		lines = theFile.readlines()
# 		theFile.close()
# 		return lines
# 	else:
# 		raise "File does not exist."

    fileName = ssxPath(fileName, baseDir)	
    try:
        theFile = file(fileName)
    except IOError:
        print ("File not found: %s" % fileName)
        raise
    else:
        lines = theFile.readlines()
        theFile.close()
        return lines

class fileData:
    x, y = 0., 0.
    xlabel, ylabel, name = '', '', ''
    filename = ''

    def __init__(self, x, y, name='', xlabel='', ylabel='', filename = None):
        self.filename = filename
        self.x = x
        self.y = y
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel



def getData(fileName):
    """Reads data from file."""
    if os.path.exists(fileName):
        theFile = file(fileName)
        lines = theFile.readlines()
        theFile.close()

        dataStart, dataEnd = [], []
        starts, ends = -1, -1
        inData = False
        variables = []
        for i in xrange(len(lines)):
            line = lines[i]
            tmp = line.split()
            if tmp != [] and tmp[0] == '#d':
                if starts == -1:
                    starts = i
                    inData = True
                ends = i
            elif tmp != [] and tmp[0] == '#v':
                variables.append(tmp[1])
            else:
                if inData == True:
                    inData = False				
                    dataStart.append(starts)
                    dataEnd.append(ends)
                    starts, ends = -1, -1

    numDataSets = len(dataStart)

    d = {}
    for i in xrange(numDataSets):
        startIndex = dataStart[i]
        stopIndex = dataEnd[i] + 1
        dataRange = stopIndex - startIndex

        dataLines = lines[startIndex:stopIndex]

        x, y = zeros(dataRange, 'f'), zeros(dataRange, 'f')

        for j in xrange(len(dataLines)):
            line = dataLines[j]
            tmp = line.split()
            x[j] = float(tmp[1])
            y[j] = float(tmp[2])
        data = fileData(x,y,variables[i])
        d[data.name] = data
    return d

def makeShotList(shotDate, range):
    if shotDate[-1] != 'r':
        shotDate = shotDate + 'r'
    if len(range) == 2:
        shotStart = range[0]
        shotEnd = range[1] + 1

        shotRange = arange(shotStart, shotEnd)
        shotRange = array(shotRange, '|S3')
    else:
        shotRange = array(range, '|S3')
    shotRange = shotRange.tolist()
    shotList = [shotDate + tmp  for tmp in shotRange]
    return shotList


def convertPNGs(dir):
    """Converts all the png files in dir to jpg files."""
    os.chdir(dir)
    cmd = "for f in *png ; do convert -quality 100 $f `basename $f png`jpg; done"
    os.popen(cmd)

def removeImages(dir):
    """Removes images in a dir and then removes the dir itself."""
    #<##> I should do something about this - it will wipe any directory.  not good

    basedir, trash = os.path.split(dir)
    os.chdir(dir)
    for file in os.listdir(dir):
        os.remove(file)
    # 	for file in glob.glob(dir + '/*.png'):
    # 		os.remove(file)
    # 	for file in glob.glob(dir + '/*.jpg'):
    # 		os.remove(file)
    os.chdir(basedir)
    os.rmdir(dir)

def gzipPDFs(dir, run):
    """Gzips the pdf's in a dir."""
    basedir, trash = os.path.split(dir)
    os.chdir(basedir)
    cmd = "tar czf %s-pdf.tgz pdf/*" % (run)
    s = os.popen(cmd)
    s.close()

def makeMovie(dir, run, word = '', fps = 3, ftype = 'jpg'):
    """Makes a movie out of the jpgs in a dir."""
    fname = '../'+ run + word + '.avi'
    os.chdir(dir)
    command = ('mencoder', 'mf://*.'+ftype, '-mf', 'type='+ftype+':w=714:h=429:fps='+str(fps), '-ovc', 'lavc', '-lavcopts', 'vcodec=mjpeg', '-oac', 'copy', '-o', fname)
    s = os.spawnvp(os.P_WAIT, 'mencoder', command)

def write_data(filename, data, header = None, delimiter = '\t'):
    """Writes data to specified file.

    Input is the filename, the data in an array, and a the header as a
    string.  Data is written in columns, delimited by default by a
    tab.  One can change this by specifying the delimiter
    argument."""

    ext = ''
    dims = len(data.shape)
    filecontents = []

    if filename[-4] == '.':
        ext = filename[-4:]
        filename = filename[:-4]

    if dims == 1:
        rows = data.shape[0]
        for row in xrange(rows):
            tmpstr = str(data[row])
            filecontents.append(tmpstr)	
    elif dims == 2:
        data = data.transpose()
        rows, col = data.shape
        for row in xrange(rows):
            tmpline = []
            for col in data[row,:]:
                tmpline.append(str(col))
            filecontents.append(delimiter.join(tmpline))


    if os.path.exists(filename + ext):
        i = 1
        newfilename = filename + '-' + str(i)
        while os.path.exists(newfilename + ext):
            i = i + 1 
            newfilename = filename + '-' + str(i)
        filename = newfilename

    filename = filename + ext
    outfile = open(filename, 'w')
    if header:
        header = header + '\n'
        outfile.writelines(header)
    outfile.writelines('\n'.join(filecontents))
    outfile.close()

