from labscript import *
from labscript_devices.PulseBlaster import PulseBlaster
from labscript_devices.NI_PCIe_6363 import NI_PCIe_6363

# Device definitions:
PulseBlaster(name='pulseblaster_0', board_number=0)
# Create a clock on one of the PulseBlaster's outputs:
ClockLine(
    name='pulseblaster_pseudoclock',
    pseudoclock=pulseblaster_0.pseudoclock,
    connection='flag 1',
)
NI_PCIe_6363(
    name='ni_card_0',
    parent_device=pulseblaster_pseudoclock,
    MAX_name='ni_pcie_6363_0',
    clock_terminal='/ni_pcie_6363_0/PFI0',
)

# Channel definitions:
Shutter(name='laser_shutter', parent_device=ni_card_0, connection='port0/line13')
AnalogOut(name='quadrupole_field', parent_device=ni_card_0, connection='ao0')
AnalogOut(name='bias_x_field', parent_device=ni_card_0, connection='ao1')

# Experiment logic:
start()
t = 0

# First laser pulse at t = 1 second:
t += 1
laser_shutter.open(t)
t += 0.5
laser_shutter.close(t)
t += 0.4

t += quadrupole_field.ramp(
    t, duration=5, initial=0, final=3, samplerate=4
)  # samplerate in Hz

bias_x_field.ramp(t - 3, duration=1, initial=0, final=2.731, samplerate=8)
# t is now 6.9s, the end of the quadrupole field ramp

# Second laser pulse
t += 0.4
laser_shutter.open(t)
t += 1
bias_x_field.constant(t=0, value=0)
t += 0.5
laser_shutter.close(t)
t += 2

# Hold bias field at the value of bias_x_final_field for 2 seconds before ending the
# shot:
bias_x_field.constant(t, value=bias_x_field_value)
t += 2
stop(t)