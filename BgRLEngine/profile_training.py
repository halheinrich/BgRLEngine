"""Profile one training iteration to find the bottleneck."""
import time
import torch
import numpy as np
from engine.network import TDNetwork
from engine.game import play_game
from engine.state import BoardState, BOARD_FEATURE_SIZE
from engine.setup_generator import SetupGenerator
from training.td_trainer import td_lambda_update

dev = torch.device("cpu")
net = TDNetwork(BOARD_FEATURE_SIZE).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=0.0001)
gen = SetupGenerator()
rng = np.random.default_rng(42)

# Warmup
play_game(net, dev, starting_state=gen.generate(), rng=rng)

# Profile 20 iterations
n = 20
play_time = 0.0
update_time = 0.0

for _ in range(n):
    start = gen.generate()

    t = time.perf_counter()
    net.eval()
    record = play_game(net, dev, starting_state=start, epsilon=0.1, rng=rng)
    play_time += time.perf_counter() - t

    t = time.perf_counter()
    td_lambda_update(net, opt, record, 0.7, dev)
    update_time += time.perf_counter() - t

print(f"Over {n} games:")
print(f"  Self-play:  {play_time:.2f}s ({play_time/n:.3f}s/game)")
print(f"  TD update:  {update_time:.2f}s ({update_time/n:.3f}s/game)")
print(f"  Total:      {play_time + update_time:.2f}s ({(play_time+update_time)/n:.3f}s/game)")
print(f"  Play %:     {100*play_time/(play_time+update_time):.0f}%")
print(f"  Update %:   {100*update_time/(play_time+update_time):.0f}%")