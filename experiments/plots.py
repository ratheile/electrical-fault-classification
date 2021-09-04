import matplotlib.pyplot as plt

def plot_class_dataset(currents, voltages, y, y_true):
  fig, axes = plt.subplots(4, 1, figsize=(10, 10))
  ax = axes[0]
  ax.set_title("currents")
  ax.plot(currents[:,0], 'r', label='Ia')
  ax.plot(currents[:,1], 'g', label='Ib')
  ax.plot(currents[:,2], 'b', label='Ic')
  ax.legend(loc="upper right")

  ax = axes[1]
  ax.set_title("voltages")
  ax.plot(voltages[:,0], 'r', label='Va')
  ax.plot(voltages[:,1], 'g', label='Vb')
  ax.plot(voltages[:,2], 'b', label='Vc')
  ax.legend(loc="upper right")

  ax = axes[2]
  ax.set_title('estimated labels')
  ax.plot(y)

  ax = axes[3]
  ax.set_title('true labels')
  ax.plot(y_true)

  fig.tight_layout()
  return fig