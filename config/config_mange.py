import toml
from typing import Dict, Any

from utils.singleton import SingletonBase


class ConfigManager(SingletonBase):
    def __init__(self):
        super().__init__()
        if not hasattr(self, '_initialized'):  # 避免重复初始化
            self._initialized = True
            self.config: Dict[str, Dict[str, Any]] = {}  # 外层字典存储文件名，内层字典存储配置内容

    def load_config(self, config_path: str, file_name: str):
        """
        加载单个 TOML 配置文件，并将其存储在 self.config 中。

        :param config_path: 配置文件路径
        :param file_name: 配置文件的标识名（用于访问）
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config[file_name] = toml.load(f)

    def load_configs(self, config_files: Dict[str, str]):
        """
        加载多个 TOML 配置文件。

        :param config_files: 字典，键为文件名标识，值为配置文件路径
        """
        for file_name, config_path in config_files.items():
            self.load_config(config_path, file_name)

    def get_config(self, file_name: str, key: str) -> Any:
        """
        获取指定配置文件中的某个配置项。

        :param file_name: 配置文件的标识名
        :param key: 配置项的键
        :return: 配置项的值
        """
        if file_name not in self.config:
            raise KeyError(f"配置文件 '{file_name}' 未加载")
        return self.config[file_name].get(key)

    def get_all_configs(self, file_name: str) -> Dict[str, Any]:
        """
        获取指定配置文件的所有配置项。

        :param file_name: 配置文件的标识名
        :return: 配置文件的所有配置项
        """
        if file_name not in self.config:
            raise KeyError(f"配置文件 '{file_name}' 未加载")
        return self.config[file_name]