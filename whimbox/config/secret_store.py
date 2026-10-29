#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""敏感配置项的本地安全存储。

设计要点：
- 敏感值不写入 config.json，只写入 secrets.json；
- Windows 上使用 DPAPI（CryptProtectData）以当前用户身份加密，密文只有同一账户
  能解开，复制到其他机器/账户无效；
- DPAPI 不可用时退化为 base64 存储并打 WARN，保证功能不中断；
- 写入采用临时文件 + os.replace 原子替换，避免写一半导致凭据全丢。
"""

import base64
import ctypes
import json
import os
import tempfile
from ctypes import wintypes
from typing import Any, Dict, Optional

from whimbox.common.path_lib import CONFIG_PATH


_CRYPTPROTECT_UI_FORBIDDEN = 0x01

_logger_cache: Any = None


def _log_warning(message: str) -> None:
    global _logger_cache
    if _logger_cache is None:
        try:
            from whimbox.common.logger import logger as _logger

            _logger_cache = _logger
        except Exception:
            _logger_cache = False
    if _logger_cache:
        _logger_cache.warning(message)
    else:
        print(f"警告: {message}")


def _log_info(message: str) -> None:
    global _logger_cache
    if _logger_cache is None:
        try:
            from whimbox.common.logger import logger as _logger

            _logger_cache = _logger
        except Exception:
            _logger_cache = False
    if _logger_cache:
        _logger_cache.info(message)
    else:
        print(message)


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_available() -> bool:
    try:
        return hasattr(ctypes, "windll") and hasattr(ctypes.windll, "crypt32")
    except Exception:
        return False


def _make_blob(data: bytes) -> "_DataBlob":
    if not data:
        raise ValueError("不能加密空数据")
    buffer = ctypes.create_string_buffer(data, len(data))
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))


def _blob_to_bytes(blob: "_DataBlob") -> bytes:
    if not blob.pbData or not blob.cbData:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)


def _dpapi_encrypt(data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    blob_in = _make_blob(data)
    blob_out = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    ):
        raise OSError(f"CryptProtectData failed: {ctypes.GetLastError()}")
    try:
        return _blob_to_bytes(blob_out)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _dpapi_decrypt(data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    blob_in = _make_blob(data)
    blob_out = _DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    ):
        raise OSError(f"CryptUnprotectData failed: {ctypes.GetLastError()}")
    try:
        return _blob_to_bytes(blob_out)
    finally:
        kernel32.LocalFree(blob_out.pbData)


class SecretStore:
    """按 "Section.key" 为引用存放敏感值的小型本地存储。"""

    _instance: Optional["SecretStore"] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecretStore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.store_file = os.path.join(CONFIG_PATH, "secrets.json")
        self._data: Dict[str, Dict[str, str]] = {}
        self._dpapi_ok = _dpapi_available()
        if not self._dpapi_ok:
            _log_warning(
                "DPAPI 不可用，密钥将以 base64 明文保存到 secrets.json（仅限本机使用）"
            )
        self._load()
        self._initialized = True

    def _load(self) -> None:
        if not os.path.exists(self.store_file):
            self._data = {}
            return
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self._data = loaded if isinstance(loaded, dict) else {}
        except Exception as e:
            _log_warning(f"读取密钥文件失败，将重新开始: {e}")
            self._data = {}

    def _write(self) -> bool:
        try:
            directory = os.path.dirname(self.store_file) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".secrets-", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.store_file)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            try:
                os.chmod(self.store_file, 0o600)
            except OSError:
                pass
            return True
        except Exception as e:
            _log_warning(f"写入密钥文件失败: {e}")
            return False

    def get(self, ref: str) -> Optional[str]:
        """读取敏感值；不存在或解密失败返回 None"""
        record = self._data.get(ref)
        if not isinstance(record, dict) or "v" not in record:
            return None
        try:
            raw = base64.b64decode(record["v"])
        except Exception:
            return None
        if record.get("enc") == "dpapi":
            try:
                return _dpapi_decrypt(raw).decode("utf-8")
            except Exception as e:
                _log_warning(f"解密 {ref} 失败（可能来自其他账户/机器）: {e}")
                return None
        return raw.decode("utf-8", errors="replace")

    def set(self, ref: str, value: str) -> bool:
        """写入敏感值；value 为空时等价于删除"""
        if not ref:
            return False
        if not value:
            return self.delete(ref)
        payload = value.encode("utf-8")
        if self._dpapi_ok:
            try:
                record = {"enc": "dpapi", "v": base64.b64encode(_dpapi_encrypt(payload)).decode("ascii")}
            except Exception as e:
                _log_warning(f"加密 {ref} 失败，退化为明文存储: {e}")
                record = {"enc": "plain", "v": base64.b64encode(payload).decode("ascii")}
        else:
            record = {"enc": "plain", "v": base64.b64encode(payload).decode("ascii")}
        self._data[ref] = record
        return self._write()

    def delete(self, ref: str) -> bool:
        if ref in self._data:
            del self._data[ref]
            return self._write()
        return True

    def has(self, ref: str) -> bool:
        return ref in self._data


# 全局单例
secret_store = SecretStore()
