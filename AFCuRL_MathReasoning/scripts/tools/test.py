from datetime import datetime
import time
timestamp = time.time()
dt = datetime.fromtimestamp(timestamp)
print("可读时间：", dt.strftime("%Y-%m-%d %H:%M:%S"))