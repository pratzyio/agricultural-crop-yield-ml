import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.DataFrame({
    'crop_type': ['Wheat', 'Rice', 'Maize'] * 40,
    'rainfall': np.random.uniform(200, 1000, 120),
    'temperature': np.random.uniform(15, 35, 120),
    'humidity': np.random.uniform(40, 80, 120),
    'soil_ph': np.random.uniform(5.5, 7.5, 120),
    'crop_yield': np.random.uniform(1.5, 5.0, 120),
})

X = data.drop(columns=['crop_yield'])
y = data['crop_yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    (
        'num',
        StandardScaler(),
        ['rainfall', 'temperature', 'humidity', 'soil_ph'],
    ),
    ('cat', OneHotEncoder(drop='first'), ['crop_type']),
])

for name, model in [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
]:
  pipe = Pipeline([('prep', preprocessor), ('model', model)])
  pipe.fit(X_train, y_train)
  preds = pipe.predict(X_test)
  print(
      f'{name} | MAE: {mean_absolute_error(y_test, preds):.2f} | R2:'
      f' {r2_score(y_test, preds):.2f}'
  )
  
