(venv) John@TWK:~/Desktop/TWK2025/Hoohacks2025/yolov5 $ python3 button.py
/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/devices.py:300: PinFactoryFallback: Falling back from lgpio: No module named 'lgpio'
  warnings.warn(
/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/devices.py:300: PinFactoryFallback: Falling back from rpigpio: No module named 'RPi'
  warnings.warn(
/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/devices.py:300: PinFactoryFallback: Falling back from pigpio: No module named 'pigpio'
  warnings.warn(
/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/devices.py:297: NativePinFactoryFallback: Falling back to the experimental pin factory NativeFactory because no other pin factory could be loaded. For best results, install RPi.GPIO or pigpio. See https://gpiozero.readthedocs.io/en/stable/api_pins.html for more information.
  warnings.warn(NativePinFactoryFallback(native_fallback_message))
Traceback (most recent call last):
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/pins/native.py", line 237, in export
    result = self._exports[pin]
             ~~~~~~~~~~~~~^^^^^
KeyError: 17

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/pins/native.py", line 247, in export
    result = os.open(self.path_value(pin),
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/sys/class/gpio/gpio17/value'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/John/Desktop/TWK2025/Hoohacks2025/yolov5/button.py", line 10, in <module>
    button = Button(17, hold_time=3)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/devices.py", line 108, in __call__
    self = super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/input_devices.py", line 412, in __init__
    super().__init__(
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/mixins.py", line 417, in __init__
    super().__init__(*args, **kwargs)
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/input_devices.py", line 167, in __init__
    self.pin.edges = 'both'
    ^^^^^^^^^^^^^^
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/pins/__init__.py", line 441, in <lambda>
    lambda self, value: self._set_edges(value),
                        ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/pins/native.py", line 519, in _set_edges
    self.factory.fs.export(self._number)
  File "/home/John/Desktop/TWK2025/Hoohacks2025/venv/lib/python3.11/site-packages/gpiozero/pins/native.py", line 251, in export
    with io.open(self.path('export'), 'wb') as f:
OSError: [Errno 22] Invalid argument
