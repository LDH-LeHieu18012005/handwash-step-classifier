from pygrabber.dshow_graph import FilterGraph

def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    for idx, name in enumerate(devices):
        print(f"{idx}: {name}")
    return devices

devices = list_cameras()
