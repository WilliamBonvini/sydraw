
from abc import ABC, abstractmethod


class Maker(ABC):
    def __init__(self, coxwain):
        self.__coxwain = coxwain

    def getCoxwain(self):
        return self.__coxwain

    @abstractmethod
    def generate_dataset_given_or(self,outliers_perc):
        pass



    def start(self):
        print("ahò")
        coxwain = self.getCoxwain()
        oRateR = coxwain.getOutliersRateRange()
        for oRate in oRateR:
            self.generate_dataset_given_or(oRate)

