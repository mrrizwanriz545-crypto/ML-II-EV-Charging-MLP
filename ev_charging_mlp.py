import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

# 1. Generate 8,760 hourly EV charging station records (1 full year)
n_samples = 8760
dates = pd.date_range("2025-01-01", periods=n_samples, freq="h")
df = pd.DataFrame({"datetime": dates})
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["month"] = df["datetime"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

commuter_wave = np.exp(-0.5 * ((df["hour"] - 18.5) / 3.2) ** 2) + \
    0.6 * np.exp(-0.5 * ((df["hour"] - 9.0) / 2.5) ** 2)
weekend_factor = np.where(df["is_weekend"] == 1, 0.72, 1.0)
df["Ambient_Temp"] = 22 + 8 * np.sin(2 * np.pi * df["month"] / 12) + 4 * np.sin(
    2 * np.pi * (df["hour"] - 6) / 24) + np.random.normal(0, 1.5, n_samples)
df["Active_Chargers"] = np.clip(np.round(
    14 * commuter_wave * weekend_factor + np.random.normal(0, 1.8, n_samples)), 0, 20)
df["EV_Count"] = np.clip(df["Active_Chargers"] +
                         np.random.poisson(1.2, n_samples), 0, 25)
df["Grid_Voltage"] = 230.0 - 0.35 * df["Active_Chargers"] + \
    np.random.normal(0, 1.2, n_samples)

# Non-linear target: Hourly Power Load (kW) with thermal & fast-charge curves
thermal_load = 0.08 * ((df["Ambient_Temp"] - 22.0) **
                       2) * (df["Active_Chargers"] / 10.0)
evening_boost = np.where((df["hour"] >= 17) & (
    df["hour"] <= 21) & (df["is_weekend"] == 0), 6.5, 0.0)
df["Power_Load_kW"] = np.maximum(0, (3.4 * df["Active_Chargers"])
                                 + 0.14 * (df["Active_Chargers"] ** 1.85)
                                 + thermal_load
                                 + evening_boost
                                 + np.random.normal(0, 1.8, n_samples))

FEATURES = ["Active_Chargers", "EV_Count", "Grid_Voltage", "Ambient_Temp",
            "hour_sin", "hour_cos", "dayofweek", "is_weekend", "month"]
TARGET = "Power_Load_kW"
df.to_csv(f"{OUT}/ev_charging_dataset.csv", index=False)

# 2. Train / Validation / Test Split (70 / 15 / 15)
X, y = df[FEATURES].values, df[TARGET].values
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=SEED)

sx = StandardScaler().fit(X_tr)
sy = StandardScaler().fit(y_tr.reshape(-1, 1))
Xtr, Xva, Xte = sx.transform(X_tr), sx.transform(X_va), sx.transform(X_te)
ytr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
yva_s = sy.transform(y_va.reshape(-1, 1)).ravel()


def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": round(mean_absolute_error(y_true, y_pred), 4),
            "MSE": round(mse, 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "R2": round(r2_score(y_true, y_pred), 4)}


def inv(a):
    return sy.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()


def train_mlp(hidden=(128, 64, 32), activation="relu", lr=1e-3, epochs=120, batch=64, alpha=1e-4):
    net = MLPRegressor(hidden_layer_sizes=hidden, activation=activation, solver="adam",
                       learning_rate_init=lr, batch_size=batch, alpha=alpha,
                       random_state=SEED, max_iter=1)
    hist = {"train_loss": [], "val_loss": []}
    rng = np.random.RandomState(SEED)
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr))
        net.partial_fit(Xtr[idx], ytr_s[idx])
        hist["train_loss"].append(mean_squared_error(ytr_s, net.predict(Xtr)))
        hist["val_loss"].append(mean_squared_error(yva_s, net.predict(Xva)))
    return net, hist


# 3. Train Baseline vs Tuned MLP
lin = LinearRegression().fit(Xtr, ytr_s)
print("Linear Regression (Test):", eval_metrics(y_te, inv(lin.predict(Xte))))

net_tuned, hist = train_mlp(
    hidden=(128, 64, 32), activation="relu", lr=0.001, epochs=120)
best_ep = int(np.argmin(hist["val_loss"])) + 1
net_final, _ = train_mlp(hidden=(128, 64, 32),
                         activation="relu", lr=0.001, epochs=best_ep)
pred_te = inv(net_final.predict(Xte))
print(f"Tuned MLP (Test, Early Stop Epoch {best_ep}):", eval_metrics(
    y_te, pred_te))

# 4. Save Training Loss Plot
plt.figure(figsize=(8, 4))
plt.plot(hist["train_loss"], label="Train MSE", color="#0F766E")
plt.plot(hist["val_loss"], label="Validation MSE", color="#D97706")
plt.axvline(best_ep, ls="--", color="gray", label=f"Best Epoch ({best_ep})")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Standardized MSE")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig01_training_curves.png", dpi=200)
plt.close()

# 5. Save Confusion Matrix Plot (Fixed cmap="GnBu")


def cls(v): return np.digitize(v, [20.0, 50.0])


yc, pc = cls(y_te), cls(pred_te)
cm = confusion_matrix(yc, pc)
acc = accuracy_score(yc, pc)
pr, rc, f1, _ = precision_recall_fscore_support(yc, pc, average="macro")
print(
    f"Tier Classification -> Accuracy: {acc:.4f}, Precision: {pr:.4f}, Recall: {rc:.4f}, F1: {f1:.4f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=[
                       "Low (<20kW)", "Medium (20-50kW)", "Peak (>=50kW)"]).plot(ax=ax, cmap="GnBu", colorbar=False)
ax.set_title("EV Load Tier Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT}/fig02_confusion_matrix.png", dpi=200)
plt.close()

# 6. Interactive Viva Demonstration Mode
print("\n=== LIVE VIVA PROTOTYPE: EV STATION LOAD PREDICTOR ===")
while True:
    choice = input(
        "\nPress [Enter] to run a sample prediction or type 'q' to quit: ").strip().lower()
    if choice == "q":
        break
    try:
        chargers = float(
            input("Enter Active Chargers (0-20, e.g., 14): ") or 14)
        temp = float(input("Enter Ambient Temp °C (e.g., 31): ") or 31)
        hr = int(input("Enter Hour of Day (0-23, e.g., 18): ") or 18)
        ev_cnt = chargers + 2
        volt = 230.0 - 0.35 * chargers
        sample = np.array([[chargers, ev_cnt, volt, temp,
                            np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24), 2, 0, 7]])
        pred_kw = inv(net_final.predict(sx.transform(sample)))[0]
        tier = ["LOW", "MEDIUM", "PEAK"][cls([pred_kw])[0]]
        print(
            f"--> Predicted Station Load: {pred_kw:.2f} kW  |  Grid Dispatch Alert: [{tier} TIER]")
    except ValueError:
        print("Invalid input—please enter numeric values.")
