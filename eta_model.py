"""Order prep + delivery time prediction (quantile GBMs, composed into an ETA).
Swap `make_synthetic_data` for your warehouse query; keep the feature names."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import train_test_split

def make_synthetic_data(n=40000, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "restaurant_id": r.integers(0, 200, n),
        "hour": r.integers(10, 23, n), "dow": r.integers(0, 7, n),
        "n_items": r.integers(1, 9, n), "order_value": r.gamma(3, 8, n),
        "kitchen_queue": r.poisson(4, n),            # open orders at the restaurant now
        "rest_avg_prep_7d": r.normal(15, 4, n).clip(6),  # rolling restaurant history
        "driver_to_rest_km": r.exponential(1.5, n),
        "rest_to_cust_km": r.exponential(2.5, n) + .3,
        "driver_speed_kmh": r.normal(22, 4, n).clip(10),
        "driver_active_orders": r.integers(0, 3, n),
        "rain": r.binomial(1, .2, n),
    })
    peak = d.hour.isin([12, 13, 19, 20, 21]).astype(int)
    d["prep_min"] = (d.rest_avg_prep_7d + 1.2*d.n_items + 1.5*d.kitchen_queue
                     + 3*peak + r.normal(0, 2.5, n)).clip(3)
    d["pickup_min"] = (d.driver_to_rest_km/d.driver_speed_kmh*60*(1+.3*d.rain)
                       + 2 + r.normal(0, 1.5, n)).clip(1)
    d["dropoff_min"] = (d.rest_to_cust_km/d.driver_speed_kmh*60*(1+.3*d.rain)
                        + 3 + 4*d.driver_active_orders + r.normal(0, 2, n)).clip(2)
    return d

FEATS = ["restaurant_id","hour","dow","n_items","order_value","kitchen_queue",
         "rest_avg_prep_7d","driver_to_rest_km","rest_to_cust_km","driver_speed_kmh",
         "driver_active_orders","rain"]
TARGETS = {"prep_min": FEATS, "pickup_min": FEATS, "dropoff_min": FEATS}

class ETAModel:
    def fit(self, df):
        cat = [FEATS.index("restaurant_id")]
        self.m = {}
        for t, f in TARGETS.items():
            for q in (.5, .9):
                self.m[(t, q)] = HGB(loss="quantile", quantile=q, categorical_features=cat,
                                     max_iter=300, learning_rate=.08).fit(df[f], df[t])
        return self

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for q, tag in ((.5, "p50"), (.9, "p90")):
            p = {t: self.m[(t, q)].predict(df[f]) for t, f in TARGETS.items()}
            # driver waits if food isn't ready; food waits if driver is slow
            out[f"eta_{tag}"] = np.maximum(p["prep_min"], p["pickup_min"]) + p["dropoff_min"]
            out[f"prep_{tag}"] = p["prep_min"]
        return out

if __name__ == "__main__":
    df = make_synthetic_data()
    tr, te = train_test_split(df, test_size=.2, random_state=1)
    model = ETAModel().fit(tr)
    pred = model.predict(te)
    truth = np.maximum(te.prep_min, te.pickup_min) + te.dropoff_min
    print("ETA MAE (p50): %.2f min" % np.abs(pred.eta_p50 - truth).mean())
    print("p90 coverage : %.1f%% (target 90%%)" % (100*(truth <= pred.eta_p90).mean()))
    print("Prep MAE      : %.2f min" % np.abs(pred.prep_p50 - te.prep_min).mean())
