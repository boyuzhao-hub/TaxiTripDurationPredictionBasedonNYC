import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
import  seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression,Ridge,BayesianRidge
from sklearn.metrics import  mean_squared_error

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print('completed load')

pd.set_option('display.float_format',lambda x:'%.3f' %x)
print(train.describe())
print(test.describe())
train.info()
test.info()
print(train.duplicated().sum())
print(train.id.duplicated().sum())
print(test.id.duplicated().sum())

m = np.mean(train['trip_duration'])
v = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*v]
train = train[train['trip_duration'] >= m - 2*v]

n = 100000
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(train.pickup_longitude[:n],
            train.pickup_latitude[:n],
            alpha = 0.1)
ax1.set_title('Pickup')
ax2.scatter(train.dropoff_longitude[:n],
            train.dropoff_latitude[:n],
            alpha = 0.1)
ax2.set_title('Dropoff')
plt.savefig('1.png')

print(train.pickup_latitude.max())
print(train.pickup_latitude.min())
print(train.pickup_longitude.max())
print(train.pickup_longitude.min())
print()
print(train.dropoff_latitude.max())
print(train.dropoff_latitude.min())
print(train.dropoff_longitude.max())
print(train.dropoff_longitude.min())

max_value = 99.999
min_value = 0.001

train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(train.pickup_longitude[:n],
            train.pickup_latitude[:n],
            alpha = 0.1)
ax1.set_title('Pickup')
ax2.scatter(train.dropoff_longitude[:n],
            train.dropoff_latitude[:n],
            alpha = 0.1)
ax2.set_title('Dropoff')
plt.savefig('2.png')

df = pd.concat([train,test])
print(train.pickup_datetime.max())
print(train.pickup_datetime.min())
print()
print(test.pickup_datetime.max())
print(test.pickup_datetime.min())
print()
print(df.pickup_datetime.max())
print(df.pickup_datetime.min())

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df['pickup_minute_of_the_day'] = df.pickup_datetime.dt.hour*60 + df.pickup_datetime.dt.minute
kmeans_pickup_time = KMeans(n_clusters=24, random_state=2).fit(df.pickup_minute_of_the_day[:500000].values.reshape(-1,1))
df['kmeans_pickup_time'] = kmeans_pickup_time.predict(df.pickup_minute_of_the_day.values.reshape(-1,1))

n = 50000
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

ax1.scatter(x = df.pickup_minute_of_the_day[:n]/60,
            y = np.random.uniform(0,1, n),
            cmap = 'Set1',
            c = df.kmeans_pickup_time[:n])
ax1.set_title('KMeans Pickup Time')

ax2.scatter(x = df.pickup_minute_of_the_day[:n]/60,
            y = np.random.uniform(0,1, n),
            cmap = 'Set1',
            c = df.pickup_datetime.dt.hour[:n])
ax2.set_title('Pickup Hour')
plt.savefig('3.png')

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

plt.hist(train.trip_duration, bins='auto')
plt.xlabel('trip_duration')
plt.ylabel('frequence')
plt.show()
plt.hist(np.log(train.trip_duration), bins='auto')
plt.xlabel('trip_duration')
plt.ylabel('log of frequence')
plt.show()

plt.plot(train.groupby(train['pickup_date']).count()[['id']], 'o-', label='train')
plt.plot(test.groupby(test['pickup_date']).count()[['id']], 'o-', label='test')
plt.title('Trips over Time.')
#plt.legend(loc=0)
plt.ylabel('Trips')
plt.savefig('4.png')

plot_vendor = train.groupby('vendor_id')['trip_duration'].mean()
plt.ylim(ymin=800)
plt.ylim(ymax=840)
sns.barplot(plot_vendor.index, plot_vendor.values)
plt.title('Time per Vendor')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')

plot_passenger = train.groupby('passenger_count')['trip_duration'].mean()
plt.subplots(1, 1, figsize=(17, 10))
plt.ylim(ymin=0)
plt.ylim(ymax=1100)
plt.title('Time per person')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')
sns.barplot(plot_passenger.index, plot_passenger.values)

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['pickup_longitude'].values[:100000], train['pickup_latitude'].values[:100000]
              , color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(test['pickup_longitude'].values[:100000], test['pickup_latitude'].values[:100000],
              color='green', s=1, label='test', alpha=0.1)
fig.suptitle('Train and test area complete overlap')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                                                         train['pickup_longitude'].values,
                                                         train['dropoff_latitude'].values,
                                                         train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                                        test['dropoff_latitude'].values,
                                                        test['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                        train['pickup_longitude'].values,
                                                                        train['dropoff_latitude'].values,
                                                                        train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values,
                                                                       test['pickup_longitude'].values,
                                                                       test['dropoff_latitude'].values,
                                                                       test['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                              train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                             test['dropoff_latitude'].values, test['dropoff_longitude'].values)

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

fig,ax=plt.subplots(ncols=1,nrows=1)
ax.scatter(train.pickup_longitude.values[:500000],train.pickup_latitude.values[:500000],s=10,lw=0,
           c=train.pickup_cluster[:500000].values,cmap='autumn',alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.savefig('5.png')

train['month'] = train['pickup_datetime'].dt.month
test['month'] = test['pickup_datetime'].dt.month
train['dayofmonth'] = train['pickup_datetime'].dt.day
test['dayofmonth'] = test['pickup_datetime'].dt.day
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour
train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
train['direction'] = train[['direction']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
test['direction'] = test[['direction']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
train['distance_haversine'] = train[['distance_haversine']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
test['distance_haversine'] = test[['distance_haversine']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
train['distance_dummy_manhattan'] = train[['distance_dummy_manhattan']].apply(
    lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
test['distance_dummy_manhattan'] = test[['distance_dummy_manhattan']].apply(
    lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
train['log_trip_duration'] = train[['trip_duration']].apply(lambda x: np.log(x + 1))

vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')
vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')
passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')
passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')
cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')
month_train = pd.get_dummies(train['month'], prefix='m', prefix_sep='_')
month_test = pd.get_dummies(test['month'], prefix='m', prefix_sep='_')
dom_train = pd.get_dummies(train['dayofmonth'], prefix='dom', prefix_sep='_')
dom_test = pd.get_dummies(test['dayofmonth'], prefix='dom', prefix_sep='_')
hour_train = pd.get_dummies(train['hour'], prefix='h', prefix_sep='_')
hour_test = pd.get_dummies(test['hour'], prefix='h', prefix_sep='_')
dow_train = pd.get_dummies(train['dayofweek'], prefix='dow', prefix_sep='_')
dow_test = pd.get_dummies(test['dayofweek'], prefix='dow', prefix_sep='_')

train = train.drop(['id','vendor_id', 'passenger_count', 'store_and_fwd_flag', 'month', 'dayofmonth', 'hour', 'dayofweek'
                       , 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1)
test = test.drop(['id','vendor_id', 'passenger_count', 'store_and_fwd_flag', 'month', 'dayofmonth', 'hour', 'dayofweek'
                     , 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1)
#test_id=test['id']
train=train.drop(['dropoff_datetime','trip_duration'],axis=1)


train_master = pd.concat([train,
                          vendor_train,
                          passenger_count_train,
                          store_and_fwd_flag_train,
                          cluster_pickup_train,
                          cluster_dropoff_train,
                          month_train,
                          dom_train,
                          hour_train,
                          dow_train
                          ], axis=1)
test_master = pd.concat([test,
                         vendor_test,
                         passenger_count_test,
                         store_and_fwd_flag_test,
                         cluster_pickup_test,
                         cluster_dropoff_test,
                         month_test,
                         dom_test,
                         hour_test,
                         dow_test
                         ], axis=1)

train_master=train_master.drop(['pickup_datetime','pickup_date'],axis=1)
test_master=test_master.drop(['pickup_datetime','pickup_date'],axis=1)

train, test = train_test_split(train_master[0:1000000], test_size=0.3)

x_train = train.drop([ 'log_trip_duration'], axis=1)
y_train = train['log_trip_duration']
x_test = test.drop(['log_trip_duration'], axis=1)
y_test = test['log_trip_duration']

y_train = y_train.reset_index().drop('index', axis=1)
y_test = y_test.reset_index().drop('index', axis=1)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_test, label=y_test)
dtest = xgb.DMatrix(test_master)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9,
            'max_depth': 6, 'subsample': 0.9, 'lambda': 1., 'nthread': 4, 'booster': 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
print('model training')
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
                  maximize=False, verbose_eval=1)
print('Modeling RMSE %.5f' % model.best_score)

pred = model.predict(dtest)
pred = np.exp(pred) - 1