"""Small existing synthetic recipe demonstrating the discovery API."""
import numpy as np
from gpubma import BFGConfig, fit_bfg

def main():
    rng = np.random.default_rng(81)
    X = rng.normal(size=(40, 5)); y = rng.normal(size=40)
    result = fit_bfg(y, X, config=BFGConfig(budget_models=20, beam_width=5, device="cpu"))
    print(result.summary())
    print(result.top_models(5).to_string(index=False))
    print(result.genealogy().to_string(index=False))

if __name__ == "__main__": main()
