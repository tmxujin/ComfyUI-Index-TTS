"""
ComfyUI Enhanced Subtitle Batch Iterator - 批量输出模式
增强版字幕批处理迭代器 - 一次性输出所有片段，ComfyUI自动批处理
"""

import json
import re
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
            }
        }
    
    # 关键改变：返回类型是Python的原始类型
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("mask_index", "character_id", "start_time", "end_time", "subtitle_text", "total_iterations")
    FUNCTION = "execute_batch"
    CATEGORY = "MultiCharacter/Batch"
    
    # 关键标记：告诉ComfyUI前五个输出是列表（批处理），最后一个是单个值
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    
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

    def _parse_json_subtitle(self, text: str) -> List[Dict]:
        """解析JSON结构化字幕"""
        try:
            data = json.loads(text)
            items = data.get("segments", []) if isinstance(data, dict) else data
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return []

        if not items or not isinstance(items, list):
            print(f"❌ JSON中未找到'segments'列表或格式不正确")
            return []

        results = []
        char_to_mask = {}
        next_mask = 1
        
        for item in items:
            char_id = item.get("id", "Character")
            
            # 旁白固定为 mask_index 0
            if char_id.lower() in ["narrator", "旁白"]:
                mask_index = 0
            else:
                if char_id not in char_to_mask:
                    char_to_mask[char_id] = next_mask
                    next_mask += 1
                mask_index = char_to_mask[char_id]

            start = self._parse_time(item.get("start", "0:00.000"))
            end = self._parse_time(item.get("end", "0:00.000"))
            subtitle = item.get("字幕", item.get("text", ""))
            
            results.append({
                "character_id": char_id, "mask_index": mask_index,
                "start_time": start, "end_time": end, "subtitle": subtitle
            })
        return results

    def execute_batch(self, subtitle_text: str, subtitle_type: str = "auto", filter_narrator: bool = False):
        """批量执行 - 一次性返回所有片段"""
        parsed_data = self._parse_json_subtitle(subtitle_text)
        
        if not parsed_data:
            print("❌ 字幕解析失败或数据为空, 返回默认值。")
            return ([0], ["Error"], [0.0], [0.0], ["Parsing Failed"], 0)
        
        if filter_narrator:
            parsed_data = [seg for seg in parsed_data if seg["mask_index"] != 0]

        total = len(parsed_data)
        if total == 0:
            print("❌ 过滤后无有效片段, 返回默认值。")
            return ([0], ["Error"], [0.0], [0.0], ["No Segments Left"], 0)
            
        print(f"✅ 批量模式启动: 共解析到 {total} 个片段")

        # 使用列表推导式高效地分离数据
        mask_indices = [seg["mask_index"] for seg in parsed_data]
        character_ids = [seg["character_id"] for seg in parsed_data]
        start_times = [seg["start_time"] for seg in parsed_data]
        end_times = [seg["end_time"] for seg in parsed_data]
        subtitles = [seg["subtitle"] for seg in parsed_data]
        
        for i, seg in enumerate(parsed_data):
            print(f"  [{i+1}/{total}] 角色'{seg['character_id']}' (mask={seg['mask_index']}) | "
                  f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s | {seg['subtitle'][:30]}...")
        
        return (mask_indices, character_ids, start_times, end_times, subtitles, total)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": EnhancedSubtitleBatchIterator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedSubtitleBatchIterator": "Enhanced Subtitle Batch Iterator",
}