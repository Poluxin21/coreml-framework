import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_samples=1000, num_features=10, output_file='data.csv'):
    data = np.random.rand(num_samples, num_features)
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])
    df['label'] = np.random.randint(0, 2, size=num_samples)  # Binary labels
    df.to_csv(output_file, index=False)
    print(f"Generated synthetic data and saved to {output_file}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    generate_synthetic_data(output_file=output_path)