import pandas as pd
import matplotlib.pyplot as plt

# Re-import file
input_path = './MapData/G29_Test.csv'

# Load CSV
df = pd.read_csv(input_path)

# Scatter plot for SDC.x vs SDC.y
print(len(df["SDC.x"]))
plt.figure(figsize=(8, 6))
print(len(df["SDC.x"]))
plt.scatter(df["SDC.x"][:], df["SDC.y"][:], s=1, label='x,y')
plt.scatter(df["SDC.trmx"][:], df["SDC.trmy"][:], s=1, label='trm')
plt.scatter(df["SDC.x"][0], df["SDC.y"][0], s=10, color='blue', label='Start Point')
# plt.scatter(df["SDC.x"][8219], df["SDC.y"][8219], s=10, color='red', label='End Point')
# plt.scatter(df["SDC.trrx"], df["SDC.trry"], s=1, label='trr')
# plt.scatter(df["SDC.trmx"], df["SDC.trmy"], s=1, label='trm')
# plt.scatter(df["SDC.trlx"], df["SDC.trly"], s=1, label='trl')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.axis('equal')
plt.show()
