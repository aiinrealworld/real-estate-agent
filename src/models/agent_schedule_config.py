from datetime import time, timedelta
from pydantic import BaseModel, Field

class AgentScheduleConfig(BaseModel):
    work_start: time = Field(default=time(9, 0))
    work_end: time = Field(default=time(18, 0))
    appointment_duration: timedelta = Field(default=timedelta(hours=1))
    schedule_buffer: timedelta = Field(default=timedelta(minutes=30))
    timezone: str = Field(default="America/Chicago")