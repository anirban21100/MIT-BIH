import datetime as dt
timestamp = dt.datetime.now()

modelname = "MIT_BIH_v_" + str(timestamp.strftime("%A %d %H")) + ".h5"
# print(modelname)

dic = {1:2, 4:6}
print(dic)
print(list(dic.keys()))