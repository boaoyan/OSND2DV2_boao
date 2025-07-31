from PyQt5.QtSerialPort import QSerialPortInfo


def get_available_ports():
    """
    获取当前系统中所有可用的串口端口信息
    :return: list of str, 如 ['COM3', 'COM4']
    """
    ports = QSerialPortInfo.availablePorts()
    available = []
    for port in ports:
        info = {
            "port": port.portName(),
            "description": port.description(),
            "manufacturer": port.manufacturer(),
            "vendor_id": port.vendorIdentifier(),
            "product_id": port.productIdentifier(),
            "is_busy": port.isBusy()
        }
        available.append(info)
    return available


def print_available_ports():
    """打印可用串口列表"""
    ports = get_available_ports()
    if not ports:
        print("🔍 没有发现任何串口设备。")
        return

    print("🔍 当前可用的串口设备：")
    print("-" * 60)
    for p in ports:
        busy_status = "❌ 被占用" if p["is_busy"] else "✅ 可用"
        print(f"端口: {p['port']}")
        print(f"  描述: {p['description']}")
        print(f"  厂商: {p['manufacturer']}")
        print(f"  VID:PID: {p['vendor_id']:04X}:{p['product_id']:04X}  {busy_status}")
        print()
    print("-" * 60)