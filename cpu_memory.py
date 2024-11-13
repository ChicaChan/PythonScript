import psutil


def monitor_cpu():
    print(f'CPU Usage: {psutil.cpu_percent()}%')
    print(f'Memory Usage: {psutil.virtual_memory().percent}%')


monitor_cpu()
