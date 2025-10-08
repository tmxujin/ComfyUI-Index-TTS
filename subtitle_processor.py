"""
ComfyUI Enhanced Subtitle Batch Iterator - 批量输出模式
增强版字幕批处理迭代器 - 一次性输出所有片段，ComfyUI自动批处理
"""

import json
import re
from typing import Dict, List, Tuple, Optional


class EnhancedSubtitleBatchIterator:
    """
    增强版字幕批处理迭代器 - 批量输出模式
    
    核心变化：
    1. 一次性解析所有片段
    2. 返回列表而不是单个值
    3. ComfyUI会自动对每个元素执行一次工作流
    4. 新增total_iterations输出，方便循环节点使用
    """
    
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
            },
            "optional": {
                "narrator_as_character_0": ("BOOLEAN", {
                    "default": True
                }),
                "filter_narrator": ("BOOLEAN", {
                    "default": False,
                    "label_on": "过滤旁白",
                    "label_off": "保留旁白"
                })
            }
        }
    
    # 🔥 关键改变：返回类型全部改为列表
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("mask_index", "character_id", "start_time", "end_time", 
                    "subtitle_text", "total_iterations")
    FUNCTION = "execute_batch"
    CATEGORY = "MultiCharacter/Batch"
    
    # 🔥 关键标记：允许批量输出
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    
    def execute_batch(self, subtitle_text: str, subtitle_type: str = "auto", 
                     narrator_as_character_0: bool = True,
                     filter_narrator: bool = False):
        """批量执行 - 一次性返回所有片段"""
        
        # 1. 解析字幕
        parsed_data = self._parse_subtitle(subtitle_text, subtitle_type, narrator_as_character_0)
        
        if not parsed_data:
            print("❌ 字幕解析失败或数据为空")
            return ([0], ["0"], [0.0], [0.0], [""], 0)
        
        # 2. 过滤旁白（如果需要）
        if filter_narrator:
            parsed_data = [seg for seg in parsed_data if seg["mask_index"] != 0]
        
        total = len(parsed_data)
        print(f"✅ 批量模式启动: {total} 个片段")
        
        # 3. 提取所有数据为列表
        mask_indices = []
        character_ids = []
        start_times = []
        end_times = []
        subtitles = []
        
        for i, seg in enumerate(parsed_data):
            mask_indices.append(seg["mask_index"])
            character_ids.append(seg["character_id"])
            start_times.append(seg["start_time"])
            end_times.append(seg["end_time"])
            subtitles.append(seg["subtitle"])
            
            print(f"  [{i+1}/{total}] 角色{seg['character_id']} (mask={seg['mask_index']}) | "
                  f"{seg['start_time']:.2f}s-{seg['end_time']:.2f}s | {seg['subtitle'][:30]}...")
        
        # 4. 返回列表 + 总次数
        return (
            mask_indices,    # [1, 2, 3, 1, 2, ...]
            character_ids,   # ["Character1", "Character2", ...]
            start_times,     # [0.0, 3.5, 7.2, ...]
            end_times,       # [3.5, 7.2, 10.8, ...]
            subtitles,       # ["文本1", "文本2", ...]
            total            # 总片段数（单个整数，不是列表）
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
        if isinstance(data, list):
            items = data
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
            
            start = self._parse_time(str(start_time_str))
            end = self._parse_time(str(end_time_str))
            
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
            time_match = re.search(
                r'(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*'
                r'(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})', 
                lines[1]
            )
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
        time_str = str(time_str).strip()
        
        # 格式1: M:SS.mmm 或 MM:SS.mmm
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
        if match:
            m, s, ms = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return m * 60 + s + ms / 1000.0
        
        # 格式2: HH:MM:SS.mmm
        match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', time_str)
        if match:
            h, m, s, ms = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return h * 3600 + m * 60 + s + ms / 1000.0
        
        return 0.0


# ============================================================
# 辅助节点: 字幕文本格式化器 (保持不变)
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
# 新增节点: 批次索引生成器
# ============================================================

class BatchIndexGenerator:
    """
    批次索引生成器
    根据total_iterations生成索引列表 [0, 1, 2, 3, ...]
    用于需要显式索引的场景
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "forceInput": True
                }),
                "start_from": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000
                })
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "generate"
    CATEGORY = "MultiCharacter/Batch"
    OUTPUT_IS_LIST = (True,)
    
    def generate(self, total_iterations: int, start_from: int = 0):
        """生成索引列表"""
        indices = list(range(start_from, start_from + total_iterations))
        print(f"🔢 生成索引: {indices[:10]}{'...' if len(indices) > 10 else ''}")
        return (indices,)


# ============================================================
# ComfyUI节点注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": EnhancedSubtitleBatchIterator,
    "SubtitleTextFormatter": SubtitleTextFormatter,
    "BatchIndexGenerator": BatchIndexGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": "Enhanced Subtitle Batch Iterator (批量模式)",
    "SubtitleTextFormatter": "Subtitle Text Formatter",
    "BatchIndexGenerator": "Batch Index Generator 🔢",
}