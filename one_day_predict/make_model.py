import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

VIEW_SIZE = 10
BEFORE_DAYS = 1
import numpy as np


kodex200 = np.load('./data/kodex200.npy', allow_pickle=True).astype('float32')
print('kodex200[:3]\n', kodex200[:3])

startP = kodex200[:,0]
startP_predict = startP[-VIEW_SIZE-BEFORE_DAYS:-BEFORE_DAYS]
# startP_predict = np.append(startP_predict, 37055)
startP_predict = startP_predict.reshape(1, startP_predict.shape[0])
print('startP_predict:\n', startP_predict)
print('startP_predict.shape', startP_predict.shape)

endP = kodex200[:,3]
endP_latest = endP[-VIEW_SIZE:]
print('startP[:3]',startP[:VIEW_SIZE])
print('endP[:3]',endP[:VIEW_SIZE])


def split_x2(seq, size):
    bbb = []
    for i in range(len(seq) - size + 1):
        bbb.append(seq[i:(i+size)])
    return np.array(bbb)

startP = split_x2(startP, VIEW_SIZE)
print('after split startP[:3]\n', startP[:3])
print('after split startP.shape', startP.shape)


endP = endP[VIEW_SIZE-1:]
print('endP.shape', endP.shape)



from sklearn.model_selection import train_test_split 
startP_train, startP_test, endP_train, endP_test = train_test_split(
    startP, endP, train_size=0.8, test_size=0.2)

print('startP_train.shape', startP_train.shape)
print('startP_test.shape', startP_test.shape)


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(startP_train) # fit하고
# startP_train = scaler.transform(startP_train)
# startP_test = scaler.transform(startP_test)
# print('after scaler startP[:3]\n', startP_train[:3])



from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold

model = XGBRegressor()
model.fit(startP_train, endP_train)

acc = model.score(startP_test, endP_test)
print("acc:", acc)

scores = cross_val_score(model, startP_train, endP_train, cv=5)
print("scores:", scores)


endP_predict = np.round(model.predict(startP_predict))
# print("endP_latest\n:", endP_latest)
print("endP_predict:\n", endP_predict)

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(9,9))
# ax = fig.add_subplot()

# ax.plot(np.arange(0, VIEW_SIZE), endP_latest, color='r')
# ax.plot(np.arange(0, VIEW_SIZE), endP_predict, color='b')

# plt.show()
