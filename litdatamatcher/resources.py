"""Conservative optional admission limits with pressure/idle hysteresis."""
from dataclasses import dataclass
import psutil


@dataclass
class ResourceGovernor:
    mode: str = 'INTERACTIVE'
    idle_entry_seconds: int = 600
    minimum_free_fraction: float = .20

    def admission(self, *, idle_seconds: float | None = None,
                  available_fraction: float | None = None, cpu_count: int | None = None):
        memory = psutil.virtual_memory()
        free = memory.available / memory.total if available_fraction is None else available_fraction
        count = cpu_count or psutil.cpu_count() or 1
        # Recovery requires five extra percentage points to avoid admission oscillation.
        if free < self.minimum_free_fraction or self.mode == 'PRESSURE' and free < self.minimum_free_fraction+.05:
            self.mode = 'PRESSURE'
        elif idle_seconds is not None and idle_seconds >= self.idle_entry_seconds:
            self.mode = 'IDLE'
        else:
            self.mode = 'INTERACTIVE'
        workers = 0 if self.mode == 'PRESSURE' else max(1,int(count*(.7 if self.mode=='IDLE' else .35)))
        return {'mode':self.mode,'admit_heavy':workers>0,'cpu_workers':workers,
                'free_ram_fraction':free,'idle_sensor':'unavailable' if idle_seconds is None else 'supplied',
                'policy':'No global process priority, driver or power changes'}
