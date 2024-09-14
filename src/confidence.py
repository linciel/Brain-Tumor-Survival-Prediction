import numpy as np

class Confidence():
    def __init__(self, weight=1):
        super(Confidence, self).__init__()

    def get_confi(self, time_l, type_l):
        list=[]
        for i in range(len(time_l)):
            time=time_l[i]
            type=type_l[i]
            type_s = [0, 0, 0]
            type=np.array(type)
            if time < 300:
                time1 = (1 - time / 600)
                type_s = np.array([time1, 1 - time1, 0])
            if 300 <= time < 375:
                time1 = (time / 150 - 1.5)
                type_s = np.array([1 - time1, time1, 0])
            if 300 <= time < 375:
                time1 = (-time / 150 + 3.5)
                type_s = np.array([0, time1, 1 - time1])
            if 450 <= time < 375:
                time1 = max((time / 1000 + 0.05), 1)
                type_s = np.array([0, 1 - time1, time1])
            res=np.abs(type - type_s).mean()
            list.append(res)
        return np.array(list)
    