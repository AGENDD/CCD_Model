import numpy as np
import pandas as pd

def generate_user_item_pairs(num_users, num_items, num_pairs, output_file):
    user_item_pairs = []
    for _ in range(num_pairs):
        user = np.random.randint(0, num_users)
        item = np.random.randint(0, num_items)
        user_item_pairs.append((user, item))
    
    df = pd.DataFrame(user_item_pairs, columns=['user', 'item'])
    df.to_csv(output_file, index=False)
    print(f"Generated {num_pairs} user-item pairs and saved to {output_file}")

# 示例参数
num_users = 100
num_items = 1000
num_pairs = 10
output_file = 'user_item_pairs.csv'

generate_user_item_pairs(num_users, num_items, num_pairs, output_file)
