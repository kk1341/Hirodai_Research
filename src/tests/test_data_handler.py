import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_handler import prepare_data

class TestDataHandler(unittest.TestCase):
    def test_prepare_data_returns_market_factor(self):
        # Setup dummy data directory and files
        input_dir = "tmp_test_data"
        os.makedirs(input_dir, exist_ok=True)
        
        # Create dummy CSV
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates.strftime("%Y%m%d"),
            "RETX": np.random.randn(10),
            "sprtrn": np.random.randn(10) # Market factor
        })
        df.to_csv(os.path.join(input_dir, "stock1.csv"), index=False)
        
        try:
            # Run prepare_data
            retx, cols, market_factor = prepare_data(
                input_dir, ["stock1.csv"], "2020-01-01", "2020-01-10", method="zero"
            )
            
            # Check shapes
            self.assertEqual(retx.shape, (10, 1))
            self.assertEqual(market_factor.shape, (10, 1))
            
            # Check content matches dummy data (sprtrn)
            # Access the sprtrn from dataframe to compare
            expected_sprtrn = df["sprtrn"].values.reshape(-1, 1)
            # Depending on interpolation there might be slight diffs if dates skipped, 
            # but here dates match exactly.
            np.testing.assert_array_almost_equal(market_factor, expected_sprtrn)
            
            print("Verification Successful: Market factor extracted correctly.")
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(input_dir)

if __name__ == "__main__":
    unittest.main()
