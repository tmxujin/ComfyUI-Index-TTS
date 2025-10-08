"""
ComfyUI Enhanced Subtitle Batch Iterator - 批量输出模式
增强版字幕批处理迭代器 - 一次性输出所有片段，ComfyUI自动批处理
"""

import json
import re
import math
from typing import Dict, List, Tuple

class EnhancedSubtitleBatchIterator:
    """
    增强版字幕批处理迭代器 - 批量输出模式
    
    核心变化：
    1. 一次性解析所有片段。
    2. 返回列表，ComfyUI会自动对每个元素执行一次工作流。
    3. 新增total_iterations输出，方便连接到其他节点。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitle_text": ("STRING", {"default": "", "multiline": True}),
                "subtitle_type": (["auto", "srt", "json"], {"default": "auto"}),
            },
            "optional": {
                "filter_narrator": ("BOOLEAN", {"default": False, "label_on": "过滤旁白", "label_off": "保留旁白"}),
                "narrator_as_character_0": ("BOOLEAN", {"default": True}),
            }
        }
    
    # 关键改变：返回类型是Python的原始类型
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "FLOAT", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("mask_index", "character_id", "start_time", "end_time", "subtitle_text", "duration", "total_iterations")
    FUNCTION = "execute_batch"
    CATEGORY = "MultiCharacter/Batch"
    
    # 关键标记：告诉ComfyUI前六个输出是列表（批处理），最后一个是单个值
    OUTPUT_IS_LIST = (True, True, True, True, True, True, False)
    
    def _parse_time(self, time_str: str) -> float:
        """健壮的时间解析函数，支持多种格式"""
        time_str = str(time_str).strip()
        
        # 格式 HH:MM:SS.mmm
        match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', time_str)
        if match:
            h, m, s, ms = map(int, match.groups())
            return h * 3600 + m * 60 + s + ms / 1000.0
            
        # 格式 M:SS.mmm 或 MM:SS.mmm
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
        if match:
            m, s, ms = map(int, match.groups())
            return m * 60 + s + ms / 1000.0
            
        print(f"⚠️无法解析时间格式: '{time_str}', 返回 0.0")
        return 0.0

    def _parse_subtitle(self, text: str, subtitle_type: str, narrator_as_char0: bool = True) -> List[Dict]:
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

    def execute_batch(self, subtitle_text: str, subtitle_type: str = "auto", 
                     filter_narrator: bool = False, narrator_as_character_0: bool = True):
        """批量执行 - 一次性返回所有片段"""
        # 解析字幕
        parsed_data = self._parse_subtitle(subtitle_text, subtitle_type, narrator_as_character_0)
        
        # 过滤旁白（如果需要）
        if filter_narrator:
            parsed_data = [seg for seg in parsed_data if seg["mask_index"] != 0]

        total = len(parsed_data)
        if total == 0:
            print("❌ 字幕解析失败或过滤后无有效片段, 返回默认值。")
            return ([0], ["Error"], [0.0], [0.0], ["No Valid Segments"], [0.0], 0)
            
        print(f"✅ 批量模式启动: 共解析到 {total} 个片段")

        # 使用列表推导式高效地分离数据
        mask_indices = [seg["mask_index"] for seg in parsed_data]
        character_ids = [seg["character_id"] for seg in parsed_data]
        start_times = [seg["start_time"] for seg in parsed_data]
        end_times = [seg["end_time"] for seg in parsed_data]
        subtitles = [seg["subtitle"] for seg in parsed_data]
        # 计算时长（结束时间减去开始时间，精确到小数点后1位）
        # 采用"只入不舍"的方式处理时长，确保视频长度足够
        durations = []
        for seg in parsed_data:
            duration = seg["end_time"] - seg["start_time"]
            if duration <= 0:
                rounded_duration = 0.0
            else:
                # 使用向上取整到0.1的倍数
                rounded_duration = math.ceil(duration * 10) / 10
            durations.append(rounded_duration)
        
        for i, seg in enumerate(parsed_data):
            duration = seg["end_time"] - seg["start_time"]
            print(f"  [{i+1}/{total}] 角色'{seg['character_id']}' (mask={seg['mask_index']}) | "
                  f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s (时长: {duration:.1f}s) | {seg['subtitle'][:30]}...")
        
        return (mask_indices, character_ids, start_times, end_times, subtitles, durations, total)


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
                "subtitle_text": ("STRING", {"forceInput": True}),
                "start_time": ("FLOAT", {"forceInput": True}),
                "end_time": ("FLOAT", {"forceInput": True}),
                "include_timeline": ("BOOLEAN", {"default": True})
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


# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": EnhancedSubtitleBatchIterator,
    "SubtitleTextFormatter": SubtitleTextFormatter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": "Enhanced Subtitle Batch Iterator",
    "SubtitleTextFormatter": "Subtitle Text Formatter",
}