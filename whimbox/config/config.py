import os
import configparser
import shutil
import tempfile
from typing import Any
import json
from whimbox.common.path_lib import CONFIG_PATH, ASSETS_PATH
from whimbox.config.default_config import DEFAULT_CONFIG, get_default_value
from whimbox.config.secret_store import secret_store


# 这些键名的配置值属于敏感信息，不写入 config.json，改存到本地加密密钥文件
SECRET_KEY_NAMES = {"api_key", "bot_token", "token", "secret", "password", "access_token"}


def _is_secret_key(key: str) -> bool:
    return isinstance(key, str) and key.lower() in SECRET_KEY_NAMES


class GlobalConfig:
    """
    配置管理类
    支持INI格式配置文件的读写操作
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化配置管理器
        """
        if self._initialized:
            return
        
        self.config_file = os.path.join(CONFIG_PATH, "config.json")
        self.config = {}
        self._update_user_config()
        self._load_config()
        self._migrate_secrets()
        self._initialized = True
    
    def _update_user_config(self):
        """确保配置目录存在，并检查新增的默认配置项"""
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(CONFIG_PATH, exist_ok=True)

        # 如果配置不存在，就直接复制默认配置
        if not os.path.exists(self.config_file):
            shutil.copy2(os.path.join(ASSETS_PATH, "default_config.json"), self.config_file)
        else:
            # 如果配置存在，就检查新增的默认配置项
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 旧按键名到新按键名的映射（兼容旧版本配置）
            key_name_mapping = {
                '鼠标左键': 'mouse_left',
                '鼠标右键': 'mouse_right',
                '鼠标中键': 'mouse_middle',
                '鼠标侧键1': 'mouse_x1',
                '鼠标侧键2': 'mouse_x2',
            }
            
            for section_name, section_data in DEFAULT_CONFIG.items():
                if section_name not in user_config:
                    user_config[section_name] = {}
                for key, config_item in section_data.items():
                    if key not in user_config[section_name]:
                        user_config[section_name][key] = config_item
                    else:
                        user_config[section_name][key]['description'] = config_item['description']
                        # 转换旧的按键名为新的按键名
                        if section_name == 'Keybinds' and 'value' in user_config[section_name][key]:
                            old_value = user_config[section_name][key]['value']
                            if old_value in key_name_mapping:
                                user_config[section_name][key]['value'] = key_name_mapping[old_value]
                        if section_name == 'OneDragon' and key == 'realm_target' and 'value' in user_config[section_name][key]:
                            old_value = user_config[section_name][key]['value']
                            if isinstance(old_value, str):
                                if old_value == '全部':
                                    user_config[section_name][key]['value'] = list(config_item.get('value', []))
                                elif old_value == '不做周本' or not old_value:
                                    user_config[section_name][key]['value'] = []
                                else:
                                    user_config[section_name][key]['value'] = [old_value]
                        if section_name == 'OneDragon' and key == 'auto_close_game' and 'value' in user_config[section_name][key]:
                            old_value = user_config[section_name][key]['value']
                            if not isinstance(old_value, list):
                                enabled = (
                                    old_value is True
                                    or isinstance(old_value, str)
                                    and old_value.lower() in ('true', '1', 'yes', 'on')
                                )
                                user_config[section_name][key]['value'] = (
                                    ['关闭游戏', '关闭奇想盒'] if enabled else []
                                )
            
            # 删除已经不存在的配置项
            sections_to_delete = [section_name for section_name in user_config 
                                 if section_name not in DEFAULT_CONFIG]
            for section_name in sections_to_delete:
                del user_config[section_name]
            for section_name in user_config:
                keys_to_delete = [key for key in user_config[section_name] 
                                if key not in DEFAULT_CONFIG[section_name]]
                for key in keys_to_delete:
                    del user_config[section_name][key]
            # 保存更新后的配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(user_config, f, ensure_ascii=False, indent=4)

    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"警告: 加载配置文件失败: {e}")
                self.config = {}
        else:
            self.config = {}
    
    def _get_item(self, section: str, key: str) -> Any:
        """安全地取配置项 dict，不存在返回 None"""
        try:
            return self.config[section][key]
        except (KeyError, TypeError):
            return None

    def _resolve_secret(self, item: Any, default: Any = None) -> str:
        """把 secret_ref 解析成真实值；解析不到时退回 default 或空串"""
        ref = item.get("secret_ref") if isinstance(item, dict) else None
        if not ref:
            return str(default) if default is not None else ""
        value = secret_store.get(ref)
        if value is None:
            return str(default) if default is not None else ""
        return value

    def get_raw(self, section: str, key: str) -> Any:
        """返回配置项原始 dict（value/description），secret 已解析为明文。

        仅供需要把整项展示给前端的场景（RPC 配置读取）使用，
        config.json 内仍只保存 secret_ref。
        """
        item = self._get_item(section, key)
        if item is None or not isinstance(item, dict):
            return item
        resolved = dict(item)
        if item.get("secret_ref"):
            resolved["value"] = self._resolve_secret(item, "")
            resolved.pop("secret_ref", None)
        return resolved

    def get(self, section: str, key: str, default: Any = None) -> str:
        """
        获取配置项的字符串值
        
        Args:
            section: 配置节名
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        item = self._get_item(section, key)
        if item is not None:
            if isinstance(item, dict) and item.get("secret_ref"):
                return self._resolve_secret(item, default)
            try:
                return item['value']
            except (KeyError, TypeError):
                pass
        if default is not None:
            return str(default)
        # 尝试从默认配置获取
        return get_default_value(section, key, str)
    
    def get_int(self, section: str, key: str, default: int = None) -> int:
        """
        获取配置项的整数值
        
        Args:
            section: 配置节名
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            return int(self.config[section][key]['value'])
        except Exception as e:
            if default is not None:
                return default
            # 尝试从默认配置获取
            return get_default_value(section, key, int)
    
    def get_float(self, section: str, key: str, default: float = None) -> float:
        """
        获取配置项的浮点数值
        
        Args:
            section: 配置节名
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            return float(self.config[section][key]['value'])
        except Exception as e:
            if default is not None:
                return default
            # 尝试从默认配置获取
            return get_default_value(section, key, float)
    
    def get_bool(self, section: str, key: str, default: bool = None) -> bool:
        """
        获取配置项的布尔值
        
        Args:
            section: 配置节名
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            return self.config[section][key]['value'].lower() in ('true', '1', 'yes', 'on')
        except Exception as e:
            if default is not None:
                return default
            # 尝试从默认配置获取
            return get_default_value(section, key, bool)
    

    def set(self, section: str, key: str, value: Any) -> None:
        """设置配置项的值"""
        if section not in self.config:
            self.config[section] = {}
        if key not in self.config[section]:
            self.config[section][key] = {}
        
        # 保持描述信息
        description = self.config[section][key].get('description', '')
        if not description and section in DEFAULT_CONFIG and key in DEFAULT_CONFIG[section]:
            description = DEFAULT_CONFIG[section][key].get('description', '')
        
        if _is_secret_key(key):
            # 敏感值落到本地加密存储，config.json 内只留引用
            ref = f"{section}.{key}"
            secret_store.set(ref, "" if value is None else str(value))
            self.config[section][key] = {
                'value': '',
                'secret_ref': ref,
                'description': description,
            }
        else:
            self.config[section][key] = {
                'value': value,
                'description': description
            }
        # 如果更新了键位设置，就要update一下
        if section == "Keybinds":
            from whimbox.common.keybind import keybind
            keybind.update_keybind()

    def _migrate_secrets(self) -> None:
        """把历史版本写在 config.json 里的明文密钥搬进本地加密存储"""
        migrated = []
        for section, keys in list(self.config.items()):
            if not isinstance(keys, dict):
                continue
            for key, item in list(keys.items()):
                if not isinstance(item, dict) or not _is_secret_key(key):
                    continue
                if item.get("secret_ref"):
                    continue
                plaintext = item.get("value")
                if not plaintext:
                    continue
                ref = f"{section}.{key}"
                if not secret_store.set(ref, str(plaintext)):
                    continue
                item["value"] = ""
                item["secret_ref"] = ref
                migrated.append(ref)
        if migrated:
            self.save()
            try:
                from whimbox.common.logger import logger
                logger.info(f"已将明文密钥迁移到本地加密存储: {', '.join(migrated)}")
            except Exception:
                print(f"已将明文密钥迁移到本地加密存储: {', '.join(migrated)}")

    def save(self) -> bool:
        """保存配置到文件（原子替换，失败不再静默）"""
        tmp_path = None
        try:
            directory = os.path.dirname(self.config_file) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=directory)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            os.replace(tmp_path, self.config_file)
            return True
        except Exception as e:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            try:
                from whimbox.common.logger import logger
                logger.exception(f"保存配置文件失败: {e}")
            except Exception:
                print(f"错误: 保存配置文件失败: {e}")
            return False

    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()

# 创建全局配置实例
global_config = GlobalConfig()

if __name__ == "__main__":
    print(global_config.get("OneDragon", "energy_cost"))
