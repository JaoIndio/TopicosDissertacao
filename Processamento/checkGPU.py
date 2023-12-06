import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
  print("{:<10} {:<15} {:<10} {:<10.2f} {:<10.2f}".format(gpu.id, gpu.name, gpu.driver, gpu.memoryFree, gpu.memoryTotal))
