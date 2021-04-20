from sklearn.datasets import load_diabetes

diab_dt = load_diabetes()

print(diab_dt.data.shape)
print(diab_dt.target.shape)
