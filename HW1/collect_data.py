import torch
import numpy as np
from homework1 import Hw1Env
from multiprocessing import Process
import os

def collect_data(idx, N):
    env = Hw1Env(render_mode="offscreen")
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    
    for i in range(N):
        env.reset()
        # Get initial state
        _, img_before = env.state()
        # Choose random action
        action_id = np.random.randint(4)
        # Execute action
        env.step(action_id)
        # Get final state
        pos_after, img_after = env.state()
        
        # Store the data
        positions[i] = torch.tensor(pos_after)
        actions[i] = action_id
        imgs_before[i] = img_before
        imgs_after[i] = img_after
        
    # Create data directory if it doesn't exist
    os.makedirs('push_data', exist_ok=True)
    
    # Save the collected data
    torch.save(positions, f'push_data/positions_{idx}.pt')
    torch.save(actions, f'push_data/actions_{idx}.pt')
    torch.save(imgs_before, f'push_data/imgs_before_{idx}.pt')
    torch.save(imgs_after, f'push_data/imgs_after_{idx}.pt')

if __name__ == "__main__":
    # Use multiple processes to collect data
    processes = []
    samples_per_process = 250  # 250 samples x 4 processes = 1000 total samples
    
    for i in range(4):
        p = Process(target=collect_data, args=(i, samples_per_process))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f"Data collection complete! Total samples: {samples_per_process * 4}")