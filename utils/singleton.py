class SingletonMeta(type):
    """
    单例元类，用于创建单例类。
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # 如果该类还没有实例化过，则创建一个新的实例
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonBase(metaclass=SingletonMeta):
    """
    使用单例模式的父类。
    """
    @classmethod
    def get_instance(cls):
        """
        获取单例实例的方法。
        """
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]

    def __init__(self):
        # 初始化代码（只会执行一次）
        print("SingletonBase 初始化")