from rich.live import Live
from rich.table import Table
import time

table = Table(title="Counter")
table.add_column("Number")

with Live(table, refresh_per_second=4):
    for i in range(5):
        table.add_row(str(i))
        time.sleep(1)