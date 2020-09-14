class Singleton(type):
    _instances = {}

    def __call__(cls,*args,**kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,cls)
        return cls._instances[cls]



'''
simply a class that handles in what folder do what and all the parameters
'''
class Coxwain:
    def __init__(self, homeDir):
        self._homeDir = homeDir

    def setTestDir(self,testDir):
        self._testDir = testDir

    def getTestDir(self):
        return self._testDir

    def setTrainDir(self,trainDir):
        self._trainDir = trainDir

    def getTrainDir(self):
        return self._trainDir

    def getHomeDir(self):
        return self._homeDir

    def setNumPointsPerSample(self,numPointsPerSample):
        self._numPointsPerSample = numPointsPerSample

    def getNumPointsPerSample(self):
        return self._numPointsPerSample

    def setNumSamples(self, num_samples):
        self._numSamples = num_samples

    def getNumSamples(self):
        return self._numSamples

    def setOutliersRateRange(self, outliers_rate_range):
        self._outliersRateRange = outliers_rate_range

    def getOutliersRateRange(self):
        return self._outliersRateRange

    def setModel(self, model):
        self._model = model

    def getModel(self):
        return self._model

    def setNoisePerc(self, noise_perc):
        self._noisePerc = noise_perc

    def getNoisePerc(self):
        return self._noisePerc

    def setDest(self, dest):
        self._dest = dest

    def getDest(self):
        return self._dest


    def setCurDir(self,curDir):
        self._curDir = curDir

    def getCurDir(self):
        return self._curDir

    def setBaseDir(self, baseDir):
        self._baseDir = baseDir

    def getBaseDir(self):
        return self._baseDir






class MyCoxwain(Coxwain,metaclass=Singleton):
    def __init__(self, homeDir):
        super().__init__(homeDir)










