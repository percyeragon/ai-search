import xgboost as xgb

class XGBFusionModel:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            learning_rate=0.05,
            max_depth=8,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42
        )

    def train(self, X, y, group):
        self.model.fit(X, y, group=group, verbose=True)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        """Speichert das trainierte Modell (JSON-Format, xgboost-nativ)."""
        self.model.save_model(str(path))

    def load(self, path):
        """Lädt ein zuvor trainiertes Modell aus dem angegebenen Pfad."""
        self.model.load_model(str(path))