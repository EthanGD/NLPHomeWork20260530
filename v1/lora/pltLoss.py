import json
import matplotlib.pyplot as plt

with open("./bge-m3-finetuned/loss_history.json", "r") as f:
    losses = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss Curve", fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig("loss_curve.png", dpi=150)
plt.show()