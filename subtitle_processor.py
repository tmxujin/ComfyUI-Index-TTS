"""
ComfyUI Enhanced Subtitle Batch Iterator
增强版字幕批处理迭代器 - 支持SRT和结构化字幕
"""

import json
import re
from typing import Dict, List, Tuple, Optional


class EnhancedSubtitleBatchIterator:
    """
    增强版字幕批处理迭代器
    功能:
    1. 支持SRT格式字幕(自动解析为角色0)
    2. 支持结构化JSON字幕(支持多角色)
    3. 旁白不忽略,统一标注为角色0
    4. 输出歌词文本供后期字幕使用
    5. 整合解析和迭代功能于一体
    """
    
    # 类级别状态存储
    _state = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitle_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": False
                }),
                "subtitle_type": (["auto", "srt", "json"], {
                    "default": "auto"
                }),
                "reset_iterator": ("BOOLEAN", {
                    "default": False
                })
            },
            "optional": {
                "narrator_as_character_0": ("BOOLEAN", {
                    "default": True
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "FLOAT", "STRING", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("character_id", "mask_index", "start_time", "end_time", 
                    "subtitle_text", "current_index", "total_count", "is_finished", "debug_info")
    FUNCTION = "execute"
    CATEGORY = "MultiCharacter/Batch"
    
    def execute(self, subtitle_text: str, subtitle_type: str = "auto", 
                reset_iterator: bool = False, narrator_as_character_0: bool = True):
        """执行字幕解析和迭代"""
        
        key = id(self)
        
        # 重置或首次运行时解析字幕
        if reset_iterator or key not in self._state:
            parsed_data = self._parse_subtitle(subtitle_text, subtitle_type, narrator_as_character_0)
            self._state[key] = {
                "data": parsed_data,
                "index": 0,
                "total": len(parsed_data)
            }
            print(f"✅ 字幕解析完成: {len(parsed_data)} 个片段")
        
        state = self._state[key]
        idx = state["index"]
        total = state["total"]
        
        # 检查是否完成
        if idx >= total:
            debug = f"✅ 批处理完成: {total}/{total}"
            print(debug)
            return ("0", 0, 0.0, 0.0, "", idx, total, True, debug)
        
        # 获取当前片段
        seg = state["data"][idx]
        state["index"] += 1
        
        is_finished = (state["index"] >= total)
        
        debug = f"🎬 [{idx+1}/{total}] 角色{seg['character_id']} | {seg['start_time']:.2f}s-{seg['end_time']:.2f}s | {seg['subtitle'][:30]}..."
        print(debug)
        
        return (
            seg["character_id"],
            seg["mask_index"],
            seg["start_time"],
            seg["end_time"],
            seg["subtitle"],
            idx + 1,  # current_index (1-based)
            total,
            is_finished,
            debug
        )
    
    def _parse_subtitle(self, text: str, subtitle_type: str, narrator_as_char0: bool) -> List[Dict]:
        """解析字幕文本"""
        text = text.strip()
        
        # 自动检测类型
        if subtitle_type == "auto":
            if text.startswith('[') or text.startswith('{'):
                subtitle_type = "json"
            else:
                subtitle_type = "srt"
        
        # 根据类型解析
        if subtitle_type == "json":
            return self._parse_json_subtitle(text, narrator_as_char0)
        else:
            return self._parse_srt_subtitle(text)
    
    def _parse_json_subtitle(self, text: str, narrator_as_char0: bool) -> List[Dict]:
        """解析JSON结构化字幕"""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return []
        
        results = []
        char_to_mask = {}  # 角色ID到mask索引的映射
        next_mask = 1
        
        # 处理JSON数据，支持两种格式
        # 格式1: 直接是数组
        if isinstance(data, list):
            items = data
        # 格式2: 包含segments字段的对象
        elif isinstance(data, dict) and "segments" in data:
            items = data["segments"]
        else:
            print(f"❌ JSON格式不支持: 期望数组或包含segments字段的对象")
            return []
        
        for item in items:
            char_id = item.get("id", item.get("character_id", "Character1"))
            
            # 处理旁白
            if narrator_as_char0 and (char_id == "Narrator" or char_id == "旁白"):
                char_id = "0"
                mask_index = 0
            else:
                # 为非旁白角色分配mask索引
                if char_id not in char_to_mask:
                    char_to_mask[char_id] = next_mask
                    next_mask += 1
                mask_index = char_to_mask[char_id]
            
            # 支持多种时间字段名称
            start_time_str = item.get("start", item.get("start_time", "0:00.000"))
            end_time_str = item.get("end", item.get("end_time", "0:00.000"))
            
            start = self._parse_time(start_time_str)
            end = self._parse_time(end_time_str)
            
            # 支持多种字幕字段名称
            subtitle_text = item.get("字幕", item.get("text", item.get("subtitle", "")))
            
            results.append({
                "character_id": char_id,
                "mask_index": mask_index,
                "start_time": start,
                "end_time": end,
                "subtitle": subtitle_text
            })
        
        return results
    
    def _parse_srt_subtitle(self, text: str) -> List[Dict]:
        """解析SRT字幕文件"""
        results = []
        blocks = text.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # 解析时间轴 "00:00:13,952 --> 00:00:21,845"
            time_match = re.search(r'(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})', lines[1])
            if not time_match:
                continue
            
            sh, sm, ss, sms = map(int, time_match.groups()[:4])
            eh, em, es, ems = map(int, time_match.groups()[4:])
            
            start_time = sh * 3600 + sm * 60 + ss + sms / 1000.0
            end_time = eh * 3600 + em * 60 + es + ems / 1000.0
            
            # 字幕文本(可能多行)
            subtitle = '\n'.join(lines[2:])
            
            results.append({
                "character_id": "0",  # SRT默认角色0
                "mask_index": 0,
                "start_time": start_time,
                "end_time": end_time,
                "subtitle": subtitle
            })
        
        return results
    
    def _parse_time(self, time_str: str) -> float:
        """解析时间字符串 M:SS.mmm -> 秒数"""
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str.strip())
        if match:
            m, s, ms = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return m * 60 + s + ms / 1000.0
        return 0.0


# ============================================================
# 辅助节点: 字幕文本格式化器
# ============================================================

class SubtitleTextFormatter:
    """
    将字幕文本格式化为可用于视频叠加的格式
    支持时间轴和纯文本输出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitle_text": ("STRING", {
                    "forceInput": True
                }),
                "start_time": ("FLOAT", {
                    "forceInput": True
                }),
                "end_time": ("FLOAT", {
                    "forceInput": True
                }),
                "include_timeline": ("BOOLEAN", {
                    "default": True
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_text",)
    FUNCTION = "format"
    CATEGORY = "MultiCharacter/Batch"
    
    def format(self, subtitle_text: str, start_time: float, end_time: float, 
               include_timeline: bool = True) -> Tuple[str]:
        """格式化字幕文本"""
        if include_timeline:
            start_str = self._format_time(start_time)
            end_str = self._format_time(end_time)
            formatted = f"[{start_str} - {end_str}]\n{subtitle_text}"
        else:
            formatted = subtitle_text
        
        return (formatted,)
    
    def _format_time(self, seconds: float) -> str:
        """秒数转时间字符串 HH:MM:SS.mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ============================================================
# ComfyUI节点注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": EnhancedSubtitleBatchIterator,
    "SubtitleTextFormatter": SubtitleTextFormatter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": "Enhanced Subtitle Batch Iterator",
    "SubtitleTextFormatter": "Subtitle Text Formatter"
}