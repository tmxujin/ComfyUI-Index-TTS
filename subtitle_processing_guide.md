# 字幕处理节点使用指南

本项目新增了两个用于字幕处理的节点，支持SRT和结构化JSON格式的字幕处理，特别适用于数字人多角色对话场景。

## 节点介绍

### 1. Enhanced Subtitle Batch Iterator (增强版字幕批处理迭代器)

这是一个功能强大的字幕处理节点，支持以下特性：

- 支持SRT格式字幕（自动解析为角色0）
- 支持结构化JSON字幕（支持多角色）
- 旁白自动标注为角色0
- 输出字幕文本供后期使用
- 整合解析和迭代功能于一体

#### 输入参数

- `subtitle_text`: 字幕文本内容（STRING类型）
- `subtitle_type`: 字幕类型（"auto", "srt", "json"，默认为"auto"）
- `reset_iterator`: 是否重置迭代器（BOOLEAN类型，默认为False）
- `narrator_as_character_0`: 是否将旁白标记为角色0（BOOLEAN类型，默认为True）

#### 输出参数

- `character_id`: 角色ID（STRING类型）
- `mask_index`: 角色对应的mask索引（INT类型）
- `start_time`: 字幕开始时间（FLOAT类型）
- `end_time`: 字幕结束时间（FLOAT类型）
- `subtitle_text`: 字幕文本内容（STRING类型）
- `current_index`: 当前索引（INT类型）
- `total_count`: 总字幕条数（INT类型）
- `is_finished`: 是否处理完成（BOOLEAN类型）
- `debug_info`: 调试信息（STRING类型）

#### 使用示例

##### JSON格式字幕示例：

```json
{
  "segments": [
    {"id": "Narrator", "start": "0:00.000", "end": "0:05.000", "字幕": "这是一个旁白片段"},
    {"id": "Character1", "start": "0:05.000", "end": "0:10.000", "字幕": "这是角色1的台词"},
    {"id": "Character2", "start": "0:10.000", "end": "0:15.000", "字幕": "这是角色2的台词"}
  ]
}
```

##### SRT格式字幕示例：

```
1
00:00:00,000 --> 00:00:05,000
这是一个旁白片段

2
00:00:05,000 --> 00:00:10,000
这是角色1的台词

3
00:00:10,000 --> 00:00:15,000
这是角色2的台词
```

### 2. Subtitle Text Formatter (字幕文本格式化器)

用于将字幕文本格式化为可用于视频叠加的格式。

#### 输入参数

- `subtitle_text`: 字幕文本内容（STRING类型，强制输入）
- `start_time`: 开始时间（FLOAT类型，强制输入）
- `end_time`: 结束时间（FLOAT类型，强制输入）
- `include_timeline`: 是否包含时间轴（BOOLEAN类型，默认为True）

#### 输出参数

- `formatted_text`: 格式化后的文本（STRING类型）

## 使用场景

### 数字人多角色对话

在数字人项目中，可以通过以下步骤使用这些节点：

1. 准备结构化的字幕数据（JSON格式）
2. 使用Enhanced Subtitle Batch Iterator节点解析字幕并逐条输出
3. 根据角色ID和mask索引切换不同的数字人模型和遮罩
4. 结合TTS节点生成对应角色的语音
5. 使用Subtitle Text Formatter节点格式化字幕文本用于视频叠加

### 工作流示例

项目中提供了一个示例工作流文件：`workflow/数字人字幕处理示例.json`，展示了如何使用这些节点。

## 注意事项

1. JSON格式支持多种字段名称，如"id"和"character_id"都可识别为角色ID
2. 时间格式支持"M:SS.mmm"格式，如"1:23.456"
3. 旁白角色可以使用"Narrator"或"旁白"标识
4. 字幕文本字段支持"字幕"、"text"、"subtitle"等多种名称
5. 节点具有状态保持功能，适用于批处理场景